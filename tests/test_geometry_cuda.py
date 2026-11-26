"""CUDA geometry regression: topology, constraints, boundary and stream use."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import create_geo_pbr
import o_voxel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def sorted_result(result, grid):
    c,v,f = [x.cpu() for x in result]
    g = [grid]*3 if isinstance(grid,int) else grid
    order = torch.argsort(c.long() @ torch.tensor([g[1]*g[2],g[2],1]))
    return c[order],v[order],f[order]


@pytest.mark.parametrize('grid',[16,[19,23,27]])
@pytest.mark.parametrize('boundary_weight',[0.,.2])
def test_open_and_closed_mesh(grid,boundary_weight):
    vertices = torch.tensor([[.1,.1,.1],[.8,.2,.2],[.2,.8,.7],[.2,.2,.9]],dtype=torch.float32)
    for faces in [torch.tensor([[0,1,2]]),torch.tensor([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])]:
        kw = dict(grid_size=grid,aabb=[[0,0,0],[1,1,1]],boundary_weight=boundary_weight,regularization_weight=.01)
        cpu=sorted_result(o_voxel.convert.mesh_to_flexible_dual_grid(vertices,faces,backend='cpu',**kw),grid)
        with torch.cuda.stream(torch.cuda.Stream()):
            cuda=sorted_result(o_voxel.convert.mesh_to_flexible_dual_grid(vertices,faces,backend='cuda',**kw),grid)
        assert torch.equal(cpu[0],cuda[0])
        assert torch.equal(cpu[2],cuda[2])
        assert torch.isfinite(cuda[1]).all()
        grid_t = torch.tensor([grid]*3 if isinstance(grid,int) else grid)
        local = cuda[1]*grid_t-cuda[0]
        assert local.min()>=-1e-5 and local.max()<=1.00001
        # CPU float32 QEF cancellation is the limiting numerical accuracy.
        assert (cpu[1]-cuda[1]).abs().max()<5e-4


def test_empty_faces_and_degenerate_faces():
    v=torch.tensor([[.1,.1,.1],[.1,.1,.1],[.1,.1,.1]])
    for f in [torch.empty((0,3),dtype=torch.int32),torch.tensor([[0,1,2]])]:
        c,p,flags=o_voxel.convert.mesh_to_flexible_dual_grid(v,f,grid_size=16,aabb=[[0]*3,[1]*3],backend='cuda')
        assert c.shape==(0,3) and p.shape==(0,3) and flags.shape==(0,3)


def test_cuda_join_order_duplicates_and_empty():
    g=torch.tensor([[2,3,4],[-2,0,1],[8,9,7]],device='cuda',dtype=torch.int32)
    p=torch.tensor([[8,9,7],[2,3,4],[8,9,7],[0,0,0]],device='cuda',dtype=torch.int32)
    out=create_geo_pbr.intersect_voxels(p,{'value':torch.arange(4,device='cuda')},g,{'value':torch.arange(3,device='cuda')})
    assert out[0].cpu().tolist()==[[8,9,7],[2,3,4],[8,9,7]]
    assert out[1]['value'].cpu().tolist()==[0,1,2]
    assert out[2].cpu().tolist()==[[2,3,4],[8,9,7]]
    assert out[3]['value'].cpu().tolist()==[0,2] and out[4]==2
    empty=create_geo_pbr.intersect_voxels(p[:0],{},g,{})
    assert len(empty[0])==len(empty[2])==empty[4]==0
    with pytest.raises(ValueError,match='unique'):
        create_geo_pbr.intersect_voxels(p,{},torch.cat((g,g[:1])),{})


def test_cuda_join_int64_overflow_falls_back():
    g=torch.tensor([[-2**40,0,0],[2**40,2**30,2**30]],device='cuda')
    p=g.flip(0)
    with pytest.raises(ValueError,match='too large'):
        create_geo_pbr.intersect_voxels(p,{},g,{})


def test_invalid_and_singular_geometry():
    v=torch.tensor([[.2,.2,.2],[.8,.2,.2],[.2,.8,.2]])
    f=torch.tensor([[0,1,2]])
    kw=dict(grid_size=16,aabb=[[0]*3,[1]*3],backend='cuda')
    c,p,_=o_voxel.convert.mesh_to_flexible_dual_grid(v,f,boundary_weight=0,regularization_weight=0,**kw)
    assert torch.isfinite(p).all()
    assert ((p*16-c)>=-1e-5).all() and ((p*16-c)<=1.00001).all()
    with pytest.raises(ValueError,match='Face index'):
        o_voxel.convert.mesh_to_flexible_dual_grid(v,torch.tensor([[0,1,3]]),**kw)
    with pytest.raises(ValueError,match='finite'):
        o_voxel.convert.mesh_to_flexible_dual_grid(v,f,boundary_weight=float('nan'),**kw)


def test_hash_capacity_retry_and_triangle_soup():
    # One large projected triangle exceeds the initial 4096 hash slots.
    # Follow with a soup containing repeated coordinates and open boundaries.
    torch.manual_seed(31)
    examples=[(torch.tensor([[.01,.01,.31],[.98,.01,.31],[.01,.98,.31]]),torch.tensor([[0,1,2]]),128)]
    vertices=torch.rand((180,3))*.8+.1
    examples.append((vertices,torch.arange(180).reshape(-1,3),16))
    for vertices,faces,grid in examples:
        kw=dict(grid_size=grid,aabb=[[0]*3,[1]*3],boundary_weight=.2,regularization_weight=.01)
        ref=sorted_result(o_voxel.convert.mesh_to_flexible_dual_grid(vertices,faces,backend='cpu',**kw),grid)
        out=sorted_result(o_voxel.convert.mesh_to_flexible_dual_grid(vertices,faces,backend='cuda',**kw),grid)
        assert torch.equal(ref[0],out[0]) and torch.equal(ref[2],out[2])
        assert torch.isfinite(out[1]).all()
        repeat=sorted_result(o_voxel.convert.mesh_to_flexible_dual_grid(vertices,faces,backend='cuda',**kw),grid)
        assert torch.equal(out[0],repeat[0]) and torch.equal(out[2],repeat[2])
        torch.testing.assert_close(out[1],repeat[1],atol=1e-6,rtol=0)
