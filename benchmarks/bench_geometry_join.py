"""Benchmark the CPU/GPU join on real geometry coordinates, without PBR sampling.

Reversing rows bypasses the existing identical-order fast path. Repeated rows
exercise multi-surface joins; absent rows ensure filtering is tested too.
"""
import json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import create_geo_pbr

torch.set_num_threads(4)
folder=Path(sys.argv[1])
c=torch.load(folder/'cuda.pt',weights_only=True)['coords']
g={'v':torch.arange(len(c),dtype=torch.int64)}
p=torch.cat((c.flip(0),c[:len(c)//10],c[:len(c)//10]+2048))
a={'v':torch.arange(len(p),dtype=torch.int64)}
torch.cuda.init()
create_geo_pbr._intersection_masks_cuda(p[:16],c[:16],torch.device('cuda'))
result={'geometry_rows':len(c),'query_rows':len(p)}
ref=None
for name,disable in [('cpu','1'),('cuda','0')]:
    os.environ['TRELLIS_DISABLE_GPU_INTERSECTION']=disable
    times=[]
    for i in range(3):
        torch.cuda.synchronize();start=time.perf_counter()
        out=create_geo_pbr.intersect_voxels(p,a,c,g)
        torch.cuda.synchronize();times.append(time.perf_counter()-start)
    result[name+'_seconds']=times
    if ref is None: ref=out
    else:
        result['exact']=all(torch.equal(x,y) for x,y in [(ref[0],out[0]),(ref[2],out[2]),(ref[1]['v'],out[1]['v']),(ref[3]['v'],out[3]['v'])]) and ref[4]==out[4]
result['speedup']=float(np.median(result['cpu_seconds'])/np.median(result['cuda_seconds']))
(folder/'join_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
