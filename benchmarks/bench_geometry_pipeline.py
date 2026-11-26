"""Measure normalization through packed CPU geometry output on existing dumps."""
import json,pickle,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import create_geo_pbr,o_voxel

torch.set_num_threads(4)
for name in sys.argv[1:]:
    folder=Path(name)
    with (folder/'optimized.pkl').open('rb') as stream: raw=pickle.load(stream)
    # Warm native imports and CUDA using a small geometry.
    o_voxel.convert.mesh_to_flexible_dual_grid(torch.tensor([[.1,.1,.1],[.8,.2,.2],[.2,.8,.7]]),
        torch.tensor([[0,1,2]]),grid_size=16,aabb=[[0]*3,[1]*3],backend='cuda')
    result={}
    for backend in ('cpu','cuda'):
        times=[]
        for rep in range(3):
            dump={**raw,'objects':[dict(obj) for obj in raw['objects']]}
            torch.cuda.synchronize();start=time.perf_counter()
            coord,attr=create_geo_pbr.convert_geometry(dump,1024,backend=backend)
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
            print(folder.name,backend,rep,times[-1],flush=True)
        result[backend+'_convert_geometry_seconds']=times
    (folder/'pipeline_results.json').write_text(json.dumps(result,indent=2)+'\n')
