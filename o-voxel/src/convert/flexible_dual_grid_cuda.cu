// CUDA implementation of the CPU flexible dual-grid algorithm.
// Scan conversion and triangle/box predicates retain the CPU arithmetic.
// QEFs are accumulated in double precision to reduce order-dependent error.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <limits>

namespace {
struct Coord { int x,y,z; __device__ int& operator[](int i) { return (&x)[i]; } };
struct Table {
    unsigned long long* keys;
    const int* rows;
    double* data; // 16 Q entries, 3 intersection sums, 1 count
    int* flags;
    int* overflow;
    long long capacity;
    int gy,gz;
    __device__ int slot(Coord c, bool insert) const {
        unsigned long long key = ((unsigned long long)c.x * gy + c.y) * gz + c.z;
        unsigned long long h=key;
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
        long long i=h & (capacity-1);
        // Bounded probing: a crowded table causes a host retry, never a hang.
        for (int probe=0; probe<256; ++probe) {
            unsigned long long old = insert ? atomicCAS(keys+i, ~0ULL, key) : keys[i];
            if (old==key || (insert && old==~0ULL)) return (int)i;
            if (!insert && old==~0ULL) return -1;
            i=(i+1)&(capacity-1);
        }
        if(insert) atomicExch(overflow,1);
        return -1;
    }
    __device__ void add(Coord c, const Eigen::Matrix4f& Q) const {
        int s=slot(c,false); if(s<0) return;
        int row=rows[s];
        for(int j=0;j<16;++j) atomicAdd(data+20LL*row+j, (double)Q.data()[j]);
    }
    __device__ void hit(Coord c, const Eigen::Matrix4f& Q, const Eigen::Vector3d& p, int flag, bool mark) const {
        int s=slot(c,mark); if(s<0 || mark) return;
        int row=rows[s];
        for(int j=0;j<16;++j) atomicAdd(data+20LL*row+j,(double)Q.data()[j]);
        for(int j=0;j<3;++j) atomicAdd(data+20LL*row+16+j,(double)(float)p[j]);
        atomicAdd(data+20LL*row+19,1.0);
        atomicOr(flags+row,flag);
    }
};
template <typename T, typename U>
__device__ U lerp(const T& a,const T& b,const T& t,const U& va,const U& vb) {
    if(a==b) return va;
    T alpha=(t-a)/(b-a); return (1-alpha)*va+alpha*vb;
}
__device__ int clampi(int x,int lo,int hi) {return max(lo,min(x,hi));}
__device__ Eigen::Vector3f vertex(const float* v,int i) {return Eigen::Vector3f(v[3LL*i],v[3LL*i+1],v[3LL*i+2]);}

__global__ void intersect_kernel(const float* vertices,const int* faces,int count,
    Eigen::Vector3f voxel_size,Eigen::Vector3i grid_min,Eigen::Vector3i grid_max,Table table,bool mark) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=count*3LL) return;
    int i=tid/3, projection=tid%3;
    Eigen::Vector3f v0=vertex(vertices,faces[3LL*i]),v1=vertex(vertices,faces[3LL*i+1]),v2=vertex(vertices,faces[3LL*i+2]);
        // Compute edge vectors and face normal
        Eigen::Vector3f e0 = v1 - v0;
        Eigen::Vector3f e1 = v2 - v1;
        Eigen::Vector3f n = e0.cross(e1).normalized();
        Eigen::Vector4f plane;
        plane << n.x(), n.y(), n.z(), -n.dot(v0);
        auto Q = plane * plane.transpose();

        // Scan-line algorithm to find intersections with the voxel grid from three directions
        /*
          t0
          | \
          |  t1
          | /
          t2
         */
        auto scan_line_fill = [&] (const int ax2) {
            int ax0 = (ax2 + 1) % 3;
            int ax1 = (ax2 + 2) % 3;

            // Canonical question
            Eigen::Vector3d t[3] = {
                Eigen::Vector3d(v0[ax0], v0[ax1], v0[ax2]),
                Eigen::Vector3d(v1[ax0], v1[ax1], v1[ax2]),
                Eigen::Vector3d(v2[ax0], v2[ax1], v2[ax2])
            };
            for (int i=1;i<3;++i) for(int j=i;j>0 && t[j].y()<t[j-1].y();--j) { auto tmp=t[j]; t[j]=t[j-1]; t[j-1]=tmp; }

            // Scan-line algorithm
            int start = clampi(int(t[0].y() / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
            int mid = clampi(int(t[1].y() / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
            int end = clampi(int(t[2].y() / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);

            auto scan_line_half = [&] (const int row_start, const int row_end, const Eigen::Vector3d t0, const Eigen::Vector3d t1, const Eigen::Vector3d t2) {
            /*
             t0
             | \
             t3-t4
             |   \
             t1---t2
             */
                for (int y_idx = row_start; y_idx < row_end; ++y_idx) {
                    double y = (y_idx + 1) * voxel_size[ax1];
                    Eigen::Vector2d t3 = lerp(t0.y(), t1.y(), y, Eigen::Vector2d(t0.x(), t0.z()), Eigen::Vector2d(t1.x(), t1.z()));
                    Eigen::Vector2d t4 = lerp(t0.y(), t2.y(), y, Eigen::Vector2d(t0.x(), t0.z()), Eigen::Vector2d(t2.x(), t2.z()));
                    if (t3.x() > t4.x()) { auto tmp=t3; t3=t4; t4=tmp; }
                    int line_start = clampi(int(t3.x() / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);
                    int line_end = clampi(int(t4.x() / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);
                    for (int x_idx = line_start; x_idx < line_end; ++x_idx) {
                        double x = (x_idx + 1) * voxel_size[ax0];
                        double z = lerp(t3.x(), t4.x(), x, t3.y(), t4.y());
                        int z_idx = int(z / voxel_size[ax2]);
                        if (z_idx >= grid_min[ax2] && z_idx < grid_max[ax2]) {
                            // For 4-connected voxels
                            for (int dx = 0; dx < 2; ++dx) {
                                for (int dy = 0; dy < 2; ++dy) {
                                    Coord coord;
                                    coord[ax0] = x_idx + dx; coord[ax1] = y_idx + dy; coord[ax2] = z_idx;
                                    Eigen::Vector3d intersect;
                                    intersect[ax0] = x; intersect[ax1] = y; intersect[ax2] = z;
                                    table.hit(coord, Q, intersect, dx==0 && dy==0 ? (1<<ax2) : 0, mark);
                                }
                            }
                        }
                    }
                }
            };
            scan_line_half(start, mid, t[0], t[1], t[2]);
            scan_line_half(mid, end, t[2], t[1], t[0]);
        };
        scan_line_fill(projection);

}

__global__ void face_kernel(const float* vertices,const int* faces,int count,
    Eigen::Vector3f voxel_size,Eigen::Vector3i grid_min,Eigen::Vector3i grid_max,Table table) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count) return;
    Eigen::Vector3f v0=vertex(vertices,faces[3LL*i]),v1=vertex(vertices,faces[3LL*i+1]),v2=vertex(vertices,faces[3LL*i+2]);
        // Compute edge vectors and face normal
        Eigen::Vector3f e0 = v1 - v0;
        Eigen::Vector3f e1 = v2 - v1;
        Eigen::Vector3f e2 = v0 - v2;
        Eigen::Vector3f n = e0.cross(e1).normalized();
        Eigen::Vector4f plane;
        plane << n.x(), n.y(), n.z(), -n.dot(v0);
        auto Q = plane * plane.transpose();

        // Compute triangle bounding box in voxel coordinates
        Eigen::Vector3f bb_min_f = v0.cwiseMin(v1).cwiseMin(v2).cwiseQuotient(voxel_size);
        Eigen::Vector3f bb_max_f = v0.cwiseMax(v1).cwiseMax(v2).cwiseQuotient(voxel_size);

        Eigen::Vector3i bb_min(max(static_cast<int>(bb_min_f.x()), grid_min.x()),
                               max(static_cast<int>(bb_min_f.y()), grid_min.y()),
                               max(static_cast<int>(bb_min_f.z()), grid_min.z()));
        Eigen::Vector3i bb_max(min(static_cast<int>(bb_max_f.x() + 1), grid_max.x()),
                               min(static_cast<int>(bb_max_f.y() + 1), grid_max.y()),
                               min(static_cast<int>(bb_max_f.z() + 1), grid_max.z()));

        // Plane test setup
        Eigen::Vector3f c(
            n.x() > 0.0f ? voxel_size.x() : 0.0f,
            n.y() > 0.0f ? voxel_size.y() : 0.0f,
            n.z() > 0.0f ? voxel_size.z() : 0.0f
        );
        float d1 = n.dot(c - v0);
        float d2 = n.dot(voxel_size - c - v0);

        // XY plane projection test setup
        int mul_xy = (n.z() < 0.0f) ? -1 : 1;
        Eigen::Vector2f n_xy_e0(-mul_xy * e0.y(), mul_xy * e0.x());
        Eigen::Vector2f n_xy_e1(-mul_xy * e1.y(), mul_xy * e1.x());
        Eigen::Vector2f n_xy_e2(-mul_xy * e2.y(), mul_xy * e2.x());

        float d_xy_e0 = -n_xy_e0.dot(v0.head<2>()) + n_xy_e0.cwiseMax(0.0f).dot(voxel_size.head<2>());
        float d_xy_e1 = -n_xy_e1.dot(v1.head<2>()) + n_xy_e1.cwiseMax(0.0f).dot(voxel_size.head<2>());
        float d_xy_e2 = -n_xy_e2.dot(v2.head<2>()) + n_xy_e2.cwiseMax(0.0f).dot(voxel_size.head<2>());

        // YZ plane projection test setup
        int mul_yz = (n.x() < 0.0f) ? -1 : 1;
        Eigen::Vector2f n_yz_e0(-mul_yz * e0.z(), mul_yz * e0.y());
        Eigen::Vector2f n_yz_e1(-mul_yz * e1.z(), mul_yz * e1.y());
        Eigen::Vector2f n_yz_e2(-mul_yz * e2.z(), mul_yz * e2.y());

        float d_yz_e0 = -n_yz_e0.dot(Eigen::Vector2f(v0.y(), v0.z())) + n_yz_e0.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.y(), voxel_size.z()));
        float d_yz_e1 = -n_yz_e1.dot(Eigen::Vector2f(v1.y(), v1.z())) + n_yz_e1.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.y(), voxel_size.z()));
        float d_yz_e2 = -n_yz_e2.dot(Eigen::Vector2f(v2.y(), v2.z())) + n_yz_e2.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.y(), voxel_size.z()));

        // ZX plane projection test setup
        int mul_zx = (n.y() < 0.0f) ? -1 : 1;
        Eigen::Vector2f n_zx_e0(-mul_zx * e0.x(), mul_zx * e0.z());
        Eigen::Vector2f n_zx_e1(-mul_zx * e1.x(), mul_zx * e1.z());
        Eigen::Vector2f n_zx_e2(-mul_zx * e2.x(), mul_zx * e2.z());

        float d_zx_e0 = -n_zx_e0.dot(Eigen::Vector2f(v0.z(), v0.x())) + n_zx_e0.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.z(), voxel_size.x()));
        float d_zx_e1 = -n_zx_e1.dot(Eigen::Vector2f(v1.z(), v1.x())) + n_zx_e1.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.z(), voxel_size.x()));
        float d_zx_e2 = -n_zx_e2.dot(Eigen::Vector2f(v2.z(), v2.x())) + n_zx_e2.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.z(), voxel_size.x()));

        // Loop over candidate voxels inside bounding box
        for (int z = bb_min.z(); z < bb_max.z(); ++z) {
            for (int y = bb_min.y(); y < bb_max.y(); ++y) {
                for (int x = bb_min.x(); x < bb_max.x(); ++x) {
                    // Voxel center
                    Eigen::Vector3f p = voxel_size.cwiseProduct(Eigen::Vector3f(x, y, z));

                    // Plane through box test
                    float nDOTp = n.dot(p);
                    if (((nDOTp + d1) * (nDOTp + d2)) > 0.0f) continue;

                    // XY projection test
                    Eigen::Vector2f p_xy(p.x(), p.y());
                    if (n_xy_e0.dot(p_xy) + d_xy_e0 < 0) continue;
                    if (n_xy_e1.dot(p_xy) + d_xy_e1 < 0) continue;
                    if (n_xy_e2.dot(p_xy) + d_xy_e2 < 0) continue;

                    // YZ projection test
                    Eigen::Vector2f p_yz(p.y(), p.z());
                    if (n_yz_e0.dot(p_yz) + d_yz_e0 < 0) continue;
                    if (n_yz_e1.dot(p_yz) + d_yz_e1 < 0) continue;
                    if (n_yz_e2.dot(p_yz) + d_yz_e2 < 0) continue;

                    // ZX projection test
                    Eigen::Vector2f p_zx(p.z(), p.x());
                    if (n_zx_e0.dot(p_zx) + d_zx_e0 < 0) continue;
                    if (n_zx_e1.dot(p_zx) + d_zx_e1 < 0) continue;
                    if (n_zx_e2.dot(p_zx) + d_zx_e2 < 0) continue;

                    // Passed all tests — mark voxel
                    table.add(Coord{x,y,z}, Q);
                }
            }
        }

}

__global__ void boundary_kernel(const float* vertices,const int64_t* edges,int64_t count,
    Eigen::Vector3f voxel_size,Eigen::Vector3i grid_min,Eigen::Vector3i grid_max,Table table,float boundary_weight) {
    int64_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count) return;
    uint64_t key=edges[i];
    Eigen::Vector3f v0=vertex(vertices,key>>32),v1=vertex(vertices,key&0xffffffff);
        // Calculate the QEF for the edge (boundary) defined by v0 and v1
        Eigen::Vector3d dir(v1.x() - v0.x(), v1.y() - v0.y(), v1.z() - v0.z());
        double segment_length = dir.norm();
        if (segment_length < 1e-6) return; // Skip degenerate edges (zero-length)
        dir.normalize();  // unit direction vector

        // Projection matrix orthogonal to the direction: I - d d^T
        Eigen::Matrix3f A = Eigen::Matrix3f::Identity() - (dir * dir.transpose()).cast<float>();

        // b = -A * v0
        Eigen::Vector3f b = -A * v0;

        // c = v0^T * A * v0
        float c = v0.transpose() * A * v0;

        // Now pack this into a 4x4 QEF matrix
        Eigen::Matrix4f Q = Eigen::Matrix4f::Zero();
        Q.block<3, 3>(0, 0) = A;
        Q.block<3, 1>(0, 3) = b;
        Q.block<1, 3>(3, 0) = b.transpose();
        Q(3, 3) = c;

        // DDA Traversal logic directly inside the function

        // Starting and ending voxel coordinates
        Eigen::Vector3i v0_voxel = (v0.cwiseQuotient(voxel_size)).array().floor().cast<int>();

        // Determine step direction for each axis based on the line direction
        Eigen::Vector3i step = (dir.array() > 0).select(Eigen::Vector3i(1, 1, 1), Eigen::Vector3i(-1, -1, -1));

        Eigen::Vector3d tMax, tDelta;
        for (int axis = 0; axis < 3; ++axis) {
            if (dir[axis] == 0.0) {
                tMax[axis] = std::numeric_limits<double>::infinity();
                tDelta[axis] = std::numeric_limits<double>::infinity();
            } else {
                float voxel_border = voxel_size[axis] * (v0_voxel[axis] + (step[axis] > 0 ? 1 : 0));
                tMax[axis] = (voxel_border - v0[axis]) / dir[axis];
                tDelta[axis] = voxel_size[axis] / std::abs(dir[axis]);
            }
        }

        // Current voxel position
        Eigen::Vector3i current = v0_voxel;

        auto accumulate = [&]() {
            if ((current.array() >= grid_min.array()).all() && (current.array() < grid_max.array()).all())
                table.add(Coord{current.x(),current.y(),current.z()}, boundary_weight * Q);
        };
        accumulate();

        // Traverse the voxels
        while (true) {
            int axis;
            if (tMax.x() < tMax.y()) {
                axis = (tMax.x() < tMax.z()) ? 0 : 2;
            } else {
                axis = (tMax.y() < tMax.z()) ? 1 : 2;
            }

            if (tMax[axis] > segment_length) break;

            current[axis] += step[axis];
            tMax[axis] += tDelta[axis];

            accumulate();
        }


}

// Fixed-size complete-pivot elimination; no host-only Eigen decomposition.
// Double precision keeps the regularized systems stable at fine resolutions.
template<int N>
__device__ Eigen::Matrix<double,N,1> small_solve(Eigen::Matrix<double,N,N> a, Eigen::Matrix<double,N,1> b) {
    int perm[N]; for(int j=0;j<N;++j) perm[j]=j;
    int rank=N; double scale=0;
    for(int j=0;j<N*N;++j) scale=fmax(scale,fabs(a.data()[j]));
    for(int k=0;k<N;++k) {
        int row=k,col=k; double best=0;
        for(int i=k;i<N;++i) for(int j=k;j<N;++j) if(fabs(a(i,j))>best) { best=fabs(a(i,j)); row=i; col=j; }
        if(best<=scale*1e-14) { rank=k; break; }
        if(row!=k) { for(int j=0;j<N;++j) {double t=a(k,j);a(k,j)=a(row,j);a(row,j)=t;} double t=b[k];b[k]=b[row];b[row]=t; }
        if(col!=k) { for(int i=0;i<N;++i) {double t=a(i,k);a(i,k)=a(i,col);a(i,col)=t;} int t=perm[k];perm[k]=perm[col];perm[col]=t; }
        for(int i=k+1;i<N;++i) { double q=a(i,k)/a(k,k); for(int j=k+1;j<N;++j) a(i,j)-=q*a(k,j); b[i]-=q*b[k]; a(i,k)=0; }
    }
    Eigen::Matrix<double,N,1> x=Eigen::Matrix<double,N,1>::Zero(), out=x;
    for(int i=rank-1;i>=0;--i) {double q=b[i];for(int j=i+1;j<rank;++j) q-=a(i,j)*x[j];x[i]=q/a(i,i);}
    for(int j=0;j<N;++j) out[perm[j]]=x[j];
    return out;
}

__global__ void solve_kernel(const int* coords,const double* data,const int* flags,int count,
    Eigen::Vector3f voxel_size,double regularization_weight,float* vertices,bool* intersected) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count) return;
    Coord coord{coords[3LL*i],coords[3LL*i+1],coords[3LL*i+2]};
    Eigen::Matrix4d Q;
    for(int j=0;j<16;++j) Q.data()[j]=data[20LL*i+j];
        double min_corner[3] = {
            coord.x * voxel_size.x(),
            coord.y * voxel_size.y(),
            coord.z * voxel_size.z()
        };
        double max_corner[3] = {
            (coord.x + 1) * voxel_size.x(),
            (coord.y + 1) * voxel_size.y(),
            (coord.z + 1) * voxel_size.z()
        };

        // Add regularization term
        if (regularization_weight > 0.0f) {
            Eigen::Vector3d p(data[20LL*i+16],data[20LL*i+17],data[20LL*i+18]);
            p /= data[20LL*i+19];

            // Construct the QEF matrix for this vertex
            Eigen::Matrix4d Qreg = Eigen::Matrix4d::Zero();
            Qreg.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity();
            Qreg.block<3,1>(0,3)    = -p;
            Qreg.block<1,3>(3,0)    = -p.transpose();
            Qreg(3,3)               = p.dot(p);

            Q += regularization_weight * data[20LL*i+19] * Qreg; // Scale by regularization weight
        }

        // Solve unconstrained
        Eigen::Matrix3d A = Q.topLeftCorner<3, 3>();
        Eigen::Vector3d b = -Q.block<3, 1>(0, 3);
        Eigen::Vector3d v_new = small_solve<3>(A,b);

        if (!(
            v_new.x() >= min_corner[0] && v_new.x() <= max_corner[0] &&
            v_new.y() >= min_corner[1] && v_new.y() <= max_corner[1] &&
            v_new.z() >= min_corner[2] && v_new.z() <= max_corner[2]
        )) {
            // Starting enumeration of constraints
            double best = std::numeric_limits<double>::infinity();

            // Solve single-constraint
            auto solve_single_constraint = [&](int fixed_axis) {
                int ax1 = (fixed_axis + 1) % 3;
                int ax2 = (fixed_axis + 2) % 3;

                Eigen::Matrix2d A;
                Eigen::Matrix2d B;
                Eigen::Vector2d q, b, x;

                A << Q(ax1, ax1), Q(ax1, ax2),
                     Q(ax2, ax1), Q(ax2, ax2);
                B << Q(ax1, fixed_axis), Q(ax1, 3),
                     Q(ax2, fixed_axis), Q(ax2, 3);


                // if lower bound
                q << min_corner[fixed_axis], 1;
                b = -B * q;
                x = small_solve<2>(A,b);
                if (
                    x.x() >= min_corner[ax1] && x.x() <= max_corner[ax1] &&
                    x.y() >= min_corner[ax2] && x.y() <= max_corner[ax2]
                ) {
                    Eigen::Vector4d p;
                    p[fixed_axis] = min_corner[fixed_axis];
                    p[ax1] = x.x();
                    p[ax2] = x.y();
                    p[3] = 1.0f;
                    double err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }

                // if upper bound
                q << max_corner[fixed_axis], 1;
                b = -B * q;
                x = small_solve<2>(A,b);
                if (
                    x.x() >= min_corner[ax1] && x.x() <= max_corner[ax1] &&
                    x.y() >= min_corner[ax2] && x.y() <= max_corner[ax2]
                ) {
                    Eigen::Vector4d p;
                    p[fixed_axis] = max_corner[fixed_axis];
                    p[ax1] = x.x();
                    p[ax2] = x.y();
                    p[3] = 1.0f;
                    double err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }
            };
            solve_single_constraint(0); // fix x
            solve_single_constraint(1); // fix y
            solve_single_constraint(2); // fix z

            // Solve two-constraint
            auto solve_two_constraint = [&](int free_axis) {
                int ax1 = (free_axis + 1) % 3;
                int ax2 = (free_axis + 2) % 3;

                double a, x;
                Eigen::Vector3d b, q;

                a = Q(free_axis, free_axis);
                b << Q(free_axis, ax1), Q(free_axis, ax2), Q(free_axis, 3);

                // if lower-lower bound
                q << min_corner[ax1], min_corner[ax2], 1;
                x = -(b.dot(q)) / a;
                if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
                    Eigen::Vector4d p;
                    p[free_axis] = x;
                    p[ax1] = min_corner[ax1];
                    p[ax2] = min_corner[ax2];
                    p[3] = 1.0f;
                    double err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }

                // if lower-upper bound
                q << min_corner[ax1], max_corner[ax2], 1;
                x = -(b.dot(q)) / a;
                if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
                    Eigen::Vector4d p;
                    p[free_axis] = x;
                    p[ax1] = min_corner[ax1];
                    p[ax2] = max_corner[ax2];
                    p[3] = 1.0f;
                    double err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }

                // if upper-lower bound
                q << max_corner[ax1], min_corner[ax2], 1;
                x = -(b.dot(q)) / a;
                if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
                    Eigen::Vector4d p;
                    p[free_axis] = x;
                    p[ax1] = max_corner[ax1];
                    p[ax2] = min_corner[ax2];
                    p[3] = 1.0f;
                    double err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }

                // if upper-upper bound
                q << max_corner[ax1], max_corner[ax2], 1;
                x = -(b.dot(q)) / a;
                if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
                    Eigen::Vector4d p;
                    p[free_axis] = x;
                    p[ax1] = max_corner[ax1];
                    p[ax2] = max_corner[ax2];
                    p[3] = 1.0f;
                    double err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }
            };
            solve_two_constraint(0); // free x
            solve_two_constraint(1); // free y
            solve_two_constraint(2); // free z

            // Solve three-constraint
            for (int x_constraint = 0; x_constraint < 2; ++x_constraint) {
                for (int y_constraint = 0; y_constraint < 2; ++y_constraint) {
                    for (int z_constraint = 0; z_constraint < 2; ++z_constraint) {
                        Eigen::Vector4d p;
                        p[0] = x_constraint ? min_corner[0] : max_corner[0];
                        p[1] = y_constraint ? min_corner[1] : max_corner[1];
                        p[2] = z_constraint ? min_corner[2] : max_corner[2];
                        p[3] = 1.0f;

                        double err = p.transpose() * Q * p;
                        if (err < best) {
                            best = err;
                            v_new << p[0], p[1], p[2];
                        }
                    }
                }
            }
        }

    for(int j=0;j<3;++j) { vertices[3LL*i+j]=(float)v_new[j]; intersected[3LL*i+j]=(flags[i] & (1<<j))!=0; }
}
} // namespace

static void check_cuda(const torch::Tensor& t,torch::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==dtype,"Expected contiguous CUDA tensor with specified dtype");
}
void dual_grid_accumulate_cuda(torch::Tensor vertices,torch::Tensor faces,torch::Tensor edges,
    std::vector<double> size,std::vector<int64_t> grid,torch::Tensor keys,torch::Tensor rows,
    torch::Tensor data,torch::Tensor flags,torch::Tensor overflow,int phase,double boundary_weight) {
    check_cuda(vertices,torch::kFloat32);check_cuda(faces,torch::kInt32);
    check_cuda(keys,torch::kInt64);check_cuda(rows,torch::kInt32);
    check_cuda(data,torch::kFloat64);check_cuda(flags,torch::kInt32);check_cuda(overflow,torch::kInt32);
    check_cuda(edges,torch::kInt64);
    TORCH_CHECK(size.size()==3 && grid.size()==3,"Expected 3D grid");
    TORCH_CHECK(vertices.dim()==2 && vertices.size(1)==3 && faces.dim()==2 && faces.size(1)==3,"Invalid mesh shape");
    TORCH_CHECK(phase>=0 && phase<=3,"Invalid phase");
    TORCH_CHECK(keys.dim()==1 && keys.numel()>0 && keys.numel()<(1LL<<31) &&
                (keys.numel() & (keys.numel()-1))==0,"Hash capacity must be a power of two below 2^31");
    TORCH_CHECK(overflow.numel()==1,"Expected one overflow counter");
    TORCH_CHECK(faces.size(0)<(1LL<<29),"Too many triangles");
    if(phase!=0) {
        TORCH_CHECK(rows.numel()==keys.numel() && data.dim()==2 && data.size(1)==20 &&
                    flags.numel()==data.size(0),"Invalid sparse QEF buffers");
    }
    for(int j=0;j<3;++j) TORCH_CHECK(grid[j]>0 && grid[j]<=INT_MAX &&
                                    std::isfinite(size[j]) && size[j]>0,"Invalid grid size");
    for(auto t : {faces,edges,keys,rows,data,flags,overflow}) TORCH_CHECK(t.device()==vertices.device(),"Device mismatch");
    c10::cuda::CUDAGuard guard(vertices.device());
    auto stream=at::cuda::getCurrentCUDAStream();
    Eigen::Vector3f vs(size[0],size[1],size[2]);
    Eigen::Vector3i lo(0,0,0),hi(grid[0],grid[1],grid[2]);
    Table table{reinterpret_cast<unsigned long long*>(keys.data_ptr<int64_t>()),rows.data_ptr<int>(),data.data_ptr<double>(),flags.data_ptr<int>(),overflow.data_ptr<int>(),keys.numel(),(int)grid[1],(int)grid[2]};
    int64_t n=faces.size(0);
    if((phase==0 || phase==1) && n) intersect_kernel<<<(3*n+127)/128,128,0,stream>>>(vertices.data_ptr<float>(),faces.data_ptr<int>(),n,vs,lo,hi,table,phase==0);
    if(phase==2 && n) face_kernel<<<(n+127)/128,128,0,stream>>>(vertices.data_ptr<float>(),faces.data_ptr<int>(),n,vs,lo,hi,table);
    if(phase==3 && edges.numel()) boundary_kernel<<<(edges.numel()+127)/128,128,0,stream>>>(vertices.data_ptr<float>(),edges.data_ptr<int64_t>(),edges.numel(),vs,lo,hi,table,boundary_weight);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::tuple<torch::Tensor,torch::Tensor> dual_grid_solve_cuda(torch::Tensor coords,torch::Tensor data,torch::Tensor flags,std::vector<double> size,double regularization_weight) {
    check_cuda(coords,torch::kInt32);check_cuda(data,torch::kFloat64);check_cuda(flags,torch::kInt32);
    TORCH_CHECK(size.size()==3 && coords.dim()==2 && coords.size(1)==3,"Invalid grid shape");
    TORCH_CHECK(data.numel()==coords.size(0)*20 && flags.numel()==coords.size(0),"Invalid QEF shape");
    TORCH_CHECK(coords.device()==data.device() && coords.device()==flags.device(),"Device mismatch");
    c10::cuda::CUDAGuard guard(coords.device());
    auto out=torch::empty(coords.sizes(),coords.options().dtype(torch::kFloat32));
    auto hits=torch::empty(coords.sizes(),coords.options().dtype(torch::kBool));
    int64_t n=coords.size(0);
    if(n) solve_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(coords.data_ptr<int>(),data.data_ptr<double>(),flags.data_ptr<int>(),n,Eigen::Vector3f(size[0],size[1],size[2]),regularization_weight,out.data_ptr<float>(),hits.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,hits};
}
