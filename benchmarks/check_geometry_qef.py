"""Audit differing CPU/CUDA dual vertices against the same accumulated QEF."""
import json,pickle,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import create_geo_pbr,o_voxel

torch.set_num_threads(4)
folder=Path(sys.argv[1]); res=1024
cpu=torch.load(folder/'cpu.pt',weights_only=True)
gpu=torch.load(folder/'cuda.pt',weights_only=True)
keys=lambda x: x.long()@torch.tensor([res**2,res,1])
order=torch.argsort(keys(cpu['coords'])); c=cpu['coords'][order]; v=cpu['vertices'][order]
assert torch.equal(c,gpu['coords'])
err=(v-gpu['vertices']).abs().amax(1)*res
sample=torch.nonzero(err>.25).flatten()
# deterministic limit, covers largest deviations as well as their population
sample=sample[torch.argsort(err[sample],descending=True)[:100000]]
original=o_voxel._C.dual_grid_solve_cuda
stats={}
def capture(coords,data,flags,size,reg):
    ids=sample.to(coords.device)
    q=data[ids,:16].reshape(-1,4,4).cpu()
    p=data[ids,16:19].cpu()/data[ids,19:20].cpu()
    weight=reg*data[ids,19].cpu()
    q[:,:3,:3]+=weight[:,None,None]*torch.eye(3,dtype=torch.double)
    q[:,:3,3]-=weight[:,None]*p
    q[:,3,:3]-=weight[:,None]*p
    q[:,3,3]+=weight*(p*p).sum(1)
    def energy(vertices):
        x=torch.cat((vertices.double(),torch.ones((len(vertices),1),dtype=torch.double)),1)
        return torch.einsum('bi,bij,bj->b',x,q,x)
    e_cpu=energy(v[sample]); e_gpu=energy(gpu['vertices'][sample])
    delta=e_cpu-e_gpu
    stats.update(sample_count=len(sample),voxels_error_over_quarter=int((err>.25).sum()),
                 cuda_energy_lower_fraction=float((delta>=-1e-12).float().mean()),
                 cpu_minus_cuda_energy_min=float(delta.min()),
                 cpu_minus_cuda_energy_mean=float(delta.mean()),
                 cpu_minus_cuda_energy_max=float(delta.max()),
                 error_voxels_over_half=int((err>.5).sum()),
                 error_voxels_max=float(err.max()))
    return original(coords,data,flags,size,reg)
o_voxel._C.dual_grid_solve_cuda=capture
with (folder/'optimized.pkl').open('rb') as f: dump=pickle.load(f)
v0,f0=create_geo_pbr._mesh_from_dump(dump)
o_voxel.convert.mesh_to_flexible_dual_grid(v0,f0,grid_size=res,aabb=[[-.5]*3,[.5]*3],boundary_weight=.2,regularization_weight=.01,backend='cuda')
(folder/'qef_audit.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2))
