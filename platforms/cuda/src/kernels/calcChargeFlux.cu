extern "C" __global__ void copyCharge(
    real4*            __restrict__  posq,
    real*             __restrict__  dedq,
    real3*            __restrict__  dqdx_val,
#ifdef USE_PBC
    const real4*      __restrict__  parameters,
    const int*        __restrict__  indexAtom
#else
    const real4*      __restrict__  parameters
#endif
){
    for (int nwork = blockIdx.x*blockDim.x+threadIdx.x; nwork < NUM_ATOMS + NUM_DQDX_PAIRS; nwork += blockDim.x*gridDim.x){
        if (nwork < NUM_ATOMS){
#ifdef USE_PBC
            posq[indexAtom[nwork]].w = parameters[nwork].x;
#else
            posq[nwork].w = parameters[nwork].x;
#endif
            dedq[nwork] = 0;
        } else {
            int ni = nwork - NUM_ATOMS;
            dqdx_val[ni] = make_real3(0);
        }
    }
}

extern "C" __global__ void calcRealCharge(
    real3*            __restrict__  dqdx_val,
    real4*            __restrict__  posq,
#ifdef USE_PBC
    const int4*       __restrict__  cf_idx,
    const real2*      __restrict__  cf_params,
    const int4*       __restrict__  wat_idx,
    const real*       __restrict__  wat_params,
    const int*        __restrict__  indexAtom,
    real4                           periodicBoxSize, 
    real4                           invPeriodicBoxSize, 
    real4                           periodicBoxVecX, 
    real4                           periodicBoxVecY, 
    real4                           periodicBoxVecZ
#else
    const int4*       __restrict__  cf_idx,
    const real2*      __restrict__  cf_params,
    const int4*       __restrict__  wat_idx,
    const real*       __restrict__  wat_params
#endif
){
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < NUM_FLUX_BONDS + NUM_FLUX_ANGLES + NUM_FLUX_WATERS; npair += blockDim.x*gridDim.x){
        if (npair < NUM_FLUX_BONDS){
            // bond
            int4 idx = cf_idx[npair]
            real2 prm = cf_params[npair];
#ifdef USE_PBC
            real4 posq1 = posq[indexAtom[idx.x]];
            real4 posq2 = posq[indexAtom[idx.y]];
            real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
            APPLY_PERIODIC_TO_DELTA(delta)
#else
            real4 posq1 = posq[idx.x];
            real4 posq2 = posq[idx.y];
            real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#endif
            real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
            real invR = RSQRT(r2);
            real r = r2 * invR;
            real dq = prm.x * (r - prm.y);
#ifdef USE_PBC
            atomicAdd(&posq[indexAtom[idx.x]].w, dq);
            atomicAdd(&posq[indexAtom[idx.y]].w, -dq);
#else
            atomicAdd(&posq[idx.x].w, dq);
            atomicAdd(&posq[idx.y].w, -dq);
#endif

            int pair1 = 4 * npair;
            int pair2 = 4 * npair + 1;
            int pair3 = 4 * npair + 2;
            int pair4 = 4 * npair + 3;
            real constant = prm.x * invR;
            real3 val = constant * delta;
            atomicAdd(&dqdx_val[pair1].x, -val.x);
            atomicAdd(&dqdx_val[pair2].x,  val.x);
            atomicAdd(&dqdx_val[pair3].x,  val.x);
            atomicAdd(&dqdx_val[pair4].x, -val.x);
            atomicAdd(&dqdx_val[pair1].y, -val.y);
            atomicAdd(&dqdx_val[pair2].y,  val.y);
            atomicAdd(&dqdx_val[pair3].y,  val.y);
            atomicAdd(&dqdx_val[pair4].y, -val.y);
            atomicAdd(&dqdx_val[pair1].z, -val.z);
            atomicAdd(&dqdx_val[pair2].z,  val.z);
            atomicAdd(&dqdx_val[pair3].z,  val.z);
            atomicAdd(&dqdx_val[pair4].z, -val.z);
        } else if (npair < NUM_FLUX_BONDS + NUM_FLUX_ANGLES) {
            // angle
            int pidx = npair - NUM_FLUX_BONDS;
            int4 idx = cf_idx[npair];
            real2 prm = cf_params[npair];
#ifdef USE_PBC
            real4 posq1 = posq[indexAtom[idx.x]];
            real4 posq2 = posq[indexAtom[idx.y]];
            real4 posq3 = posq[indexAtom[idx.z]];
            real3 d21 = make_real3(posq1.x-posq2.x, posq1.y-posq2.y, posq1.z-posq2.z);
            APPLY_PERIODIC_TO_DELTA(d21)
            real3 d23 = make_real3(posq3.x-posq2.x, posq3.y-posq2.y, posq3.z-posq2.z);
            APPLY_PERIODIC_TO_DELTA(d23)
            real3 d13 = make_real3(posq3.x-posq1.x, posq3.y-posq1.y, posq3.z-posq1.z);
            APPLY_PERIODIC_TO_DELTA(d13)
#else
            real4 posq1 = posq[idx.x];
            real4 posq2 = posq[idx.y];
            real4 posq3 = posq[idx.z];
            real3 d21 = make_real3(posq1.x-posq2.x, posq1.y-posq2.y, posq1.z-posq2.z);
            real3 d23 = make_real3(posq3.x-posq2.x, posq3.y-posq2.y, posq3.z-posq2.z);
            real3 d13 = make_real3(posq3.x-posq1.x, posq3.y-posq1.y, posq3.z-posq1.z);
#endif
            real r21_2 = d21.x*d21.x + d21.y*d21.y + d21.z*d21.z;
            real invR21 = RSQRT(r21_2);
            // real r21 = r21_2 * invR21;

            real r23_2 = d23.x*d23.x + d23.y*d23.y + d23.z*d23.z;
            real invR23 = RSQRT(r23_2);
            // real r23 = r23_2 * invR23;

            real r13_2 = d13.x*d13.x + d13.y*d13.y + d13.z*d13.z;
            // real invR13 = RSQRT(r13_2);
            // real r13 = r13_2 * invR13;

            real cost = (r23_2 + r21_2 - r13_2) * 0.5 * invR21 * invR23;
            real angle = ACOS(cost);

            real dq = prm.x * (angle - prm.y);
#ifdef USE_PBC
            atomicAdd(&posq[indexAtom[idx.x]].w, dq);
            atomicAdd(&posq[indexAtom[idx.y]].w, -2 * dq);
            atomicAdd(&posq[indexAtom[idx.z]].w, dq);
#else
            atomicAdd(&posq[idx.x].w, dq);
            atomicAdd(&posq[idx.y].w, -2 * dq);
            atomicAdd(&posq[idx.z].w, dq);
#endif

            int pair1 = BSHIFT + 9 * pidx;
            int pair2 = BSHIFT + 9 * pidx + 1;
            int pair3 = BSHIFT + 9 * pidx + 2;
            int pair4 = BSHIFT + 9 * pidx + 3;
            int pair5 = BSHIFT + 9 * pidx + 4;
            int pair6 = BSHIFT + 9 * pidx + 5;
            int pair7 = BSHIFT + 9 * pidx + 6;
            int pair8 = BSHIFT + 9 * pidx + 7;
            int pair9 = BSHIFT + 9 * pidx + 8;
            real one_const = prm.x * RSQRT(1 - cost*cost);
            real fin_const1 = invR21 * invR23 * one_const;
            real fin_const2_r21 = cost * one_const * invR21 * invR21;
            real fin_const2_r23 = cost * one_const * invR23 * invR23;

            real3 v1 = - fin_const1 * d23 + fin_const2_r21 * d21;
            real3 v3 = - fin_const1 * d21 + fin_const2_r23 * d23;
            real3 v2 = - v1 - v3;

            atomicAdd(&dqdx_val[pair1].x, v1.x);
            atomicAdd(&dqdx_val[pair2].x, v2.x);
            atomicAdd(&dqdx_val[pair3].x, v3.x);
            atomicAdd(&dqdx_val[pair4].x, -2 * v1.x);
            atomicAdd(&dqdx_val[pair5].x, -2 * v2.x);
            atomicAdd(&dqdx_val[pair6].x, -2 * v3.x);
            atomicAdd(&dqdx_val[pair7].x, v1.x);
            atomicAdd(&dqdx_val[pair8].x, v2.x);
            atomicAdd(&dqdx_val[pair9].x, v3.x);
            atomicAdd(&dqdx_val[pair1].y, v1.y);
            atomicAdd(&dqdx_val[pair2].y, v2.y);
            atomicAdd(&dqdx_val[pair3].y, v3.y);
            atomicAdd(&dqdx_val[pair4].y, -2 * v1.y);
            atomicAdd(&dqdx_val[pair5].y, -2 * v2.y);
            atomicAdd(&dqdx_val[pair6].y, -2 * v3.y);
            atomicAdd(&dqdx_val[pair7].y, v1.y);
            atomicAdd(&dqdx_val[pair8].y, v2.y);
            atomicAdd(&dqdx_val[pair9].y, v3.y);
            atomicAdd(&dqdx_val[pair1].z, v1.z);
            atomicAdd(&dqdx_val[pair2].z, v2.z);
            atomicAdd(&dqdx_val[pair3].z, v3.z);
            atomicAdd(&dqdx_val[pair4].z, -2 * v1.z);
            atomicAdd(&dqdx_val[pair5].z, -2 * v2.z);
            atomicAdd(&dqdx_val[pair6].z, -2 * v3.z);
            atomicAdd(&dqdx_val[pair7].z, v1.z);
            atomicAdd(&dqdx_val[pair8].z, v2.z);
            atomicAdd(&dqdx_val[pair9].z, v3.z);
        } else {
            int pidx = npair - NUM_FLUX_BONDS - NUM_FLUX_ANGLES;
            int4 idx = wat_idx[pidx];
            real k1 = wat_params[5*pidx];
            real k2 = wat_params[5*pidx+1];
            real kub = wat_params[5*pidx+2];
            real b0 = wat_params[5*pidx+3];
            real ub0 = wat_params[5*pidx+4];
#ifdef USE_PBC
            real4 posq1 = posq[indexAtom[idx.x]];
            real4 posq2 = posq[indexAtom[idx.y]];
            real4 posq3 = posq[indexAtom[idx.z]];
            real3 d12 = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
            APPLY_PERIODIC_TO_DELTA(d12)
            real3 d13 = make_real3(posq3.x-posq1.x, posq3.y-posq1.y, posq3.z-posq1.z);
            APPLY_PERIODIC_TO_DELTA(d13)
            real3 d23 = make_real3(posq3.x-posq2.x, posq3.y-posq2.y, posq3.z-posq2.z);
            APPLY_PERIODIC_TO_DELTA(d23)
#else
            real4 posq1 = posq[idx.x];
            real4 posq2 = posq[idx.y];
            real4 posq3 = posq[idx.z];
            real3 d12 = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
            real3 d13 = make_real3(posq3.x-posq1.x, posq3.y-posq1.y, posq3.z-posq1.z);
            real3 d23 = make_real3(posq3.x-posq2.x, posq3.y-posq2.y, posq3.z-posq2.z);
#endif
            real r12_2 = d12.x * d12.x + d12.y * d12.y + d12.z * d12.z;
            real r13_2 = d13.x * d13.x + d13.y * d13.y + d13.z * d13.z;
            real r23_2 = d23.x * d23.x + d23.y * d23.y + d23.z * d23.z;
            real invR12 = RSQRT(r12_2);
            real invR13 = RSQRT(r13_2);
            real invR23 = RSQRT(r23_2);
            real r12 = r12_2 * invR12;
            real r13 = r13_2 * invR13;
            real r23 = r23_2 * invR23;
            real3 d12_n = d12 * invR12;
            real3 d13_n = d13 * invR13;
            real3 d23_n = d23 * invR23;
            
            real dq2 = k1 * (r12 - b0) + k2 * (r13 - b0) + kub * (r23 - ub0);
            real dq3 = k1 * (r13 - b0) + k2 * (r12 - b0) + kub * (r23 - ub0);
            real dq1 = - dq2 - dq3;

#ifdef USE_PBC
            atomicAdd(&posq[indexAtom[idx.x]].w, dq1);
            atomicAdd(&posq[indexAtom[idx.y]].w, dq2);
            atomicAdd(&posq[indexAtom[idx.z]].w, dq3);
#else
            atomicAdd(&posq[idx.x].w, dq1);
            atomicAdd(&posq[idx.y].w, dq2);
            atomicAdd(&posq[idx.z].w, dq3);
#endif
            int pair1 = BASHIFT + 9 * pidx;
            int pair2 = BASHIFT + 9 * pidx + 1;
            int pair3 = BASHIFT + 9 * pidx + 2;
            int pair4 = BASHIFT + 9 * pidx + 3;
            int pair5 = BASHIFT + 9 * pidx + 4;
            int pair6 = BASHIFT + 9 * pidx + 5;
            int pair7 = BASHIFT + 9 * pidx + 6;
            int pair8 = BASHIFT + 9 * pidx + 7;
            int pair9 = BASHIFT + 9 * pidx + 8;
            real3 v5 = d12_n * k1 - d23_n * kub;
            real3 v6 = d13_n * k2 + d23_n * kub;
            real3 v8 = d12_n * k2 - d23_n * kub;
            real3 v9 = d13_n * k1 + d23_n * kub;
            real3 v7 = - v8 - v9;
            real3 v4 = - v5 - v6;
            real3 v1 = - v4 - v7;
            real3 v2 = - v5 - v8;
            real3 v3 = - v6 - v9;

            dqdx_val[pair1] += v1;
            dqdx_val[pair2] += v2;
            dqdx_val[pair3] += v3;
            dqdx_val[pair4] += v4;
            dqdx_val[pair5] += v5;
            dqdx_val[pair6] += v6;
            dqdx_val[pair7] += v7;
            dqdx_val[pair8] += v8;
            dqdx_val[pair9] += v9;
        }
    }
}

extern "C" __global__ void multdQdX(
    unsigned long long*   __restrict__    forceBuffers, 
    const real*           __restrict__    dedq,
#ifdef USE_PBC
    const int*            __restrict__    indexAtom,
#endif
    const int*            __restrict__    dqdx_dqidx,
    const int*            __restrict__    dqdx_dxidx,
    const real3*          __restrict__    dqdx_val
){
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < NUM_DQDX_PAIRS; npair += blockDim.x*gridDim.x){
#ifdef USE_PBC
        int p1 = dqdx_dqidx[npair];
        int p2 = indexAtom[dqdx_dxidx[npair]];
#else
        int p1 = dqdx_dqidx[npair];
        int p2 = dqdx_dxidx[npair];
#endif
        real3 nval = - dqdx_val[npair] * dedq[p1];
        atomicAdd(&forceBuffers[p2], static_cast<unsigned long long>((long long) (nval.x*0x100000000)));
        atomicAdd(&forceBuffers[p2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (nval.y*0x100000000)));
        atomicAdd(&forceBuffers[p2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (nval.z*0x100000000)));
    }
}

extern "C" __global__ void printdQdX(
    const int*            __restrict__    dqdx_dqidx,
    const int*            __restrict__    dqdx_dxidx,
    const real3*          __restrict__    dqdx_val
){
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < NUM_DQDX_PAIRS; npair += blockDim.x*gridDim.x){
        real3 nval = dqdx_val[npair];
        printf("%i %i %f %f %f\n", dqdx_dqidx[npair], dqdx_dxidx[npair], nval.x, nval.y, nval.z);
    }
}