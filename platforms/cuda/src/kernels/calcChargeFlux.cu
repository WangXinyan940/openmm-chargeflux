extern "C" __global__ void copyCharge(
    real4*            __restrict__  posq,
    real*             __restrict__  dedq,
    const real*       __restrict__  parameters,
#ifdef USE_PBC
    const int*        __restrict__  indexAtom
#endif
){
    for (int natom = blockIdx.x*blockDim.x+threadIdx.x; natom < NUM_ATOMS; natom += blockDim.x*gridDim.x){
#ifdef USE_PBC
        atomicAdd(&posq[indexAtom[natom]].w,parameters[natom*3]);
        print("%f %f\n", posq[indexAtom[natom]].w, parameters[natom*3]);
#else
        atomicAdd(&posq[natom].w,parameters[natom*3]);
        printf("%f %f\n", posq[natom].w, parameters[natom*3]);
#endif
        dedq[natom] = 0;
    }
}

extern "C" __global__ void calcRealCharge(
    real*             __restrict__  dqdx_val,
    real4*            __restrict__  posq,
#ifdef USE_PBC
    const int*        __restrict__  cf_idx,
    const real*       __restrict__  cf_params,
    const int*        __restrict__  indexAtom,
    real4                           periodicBoxSize, 
    real4                           invPeriodicBoxSize, 
    real4                           periodicBoxVecX, 
    real4                           periodicBoxVecY, 
    real4                           periodicBoxVecZ
#else
    const int*        __restrict__  cf_idx,
    const real*       __restrict__  cf_params
#endif
){
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < NUM_FLUX_BONDS + NUM_FLUX_ANGLES; npair += blockDim.x*gridDim.x){
        if (npair < NUM_FLUX_BONDS){
            // bond
            int idx1 = cf_idx[npair*2];
            int idx2 = cf_idx[npair*2+1];
            real k = cf_params[npair*2];
            real b = cf_params[npair*2+1];
#ifdef USE_PBC
            real4 posq1 = posq[indexAtom[idx1]];
            real4 posq2 = posq[indexAtom[idx2]];
            real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
            APPLY_PERIODIC_TO_DELTA(delta)
#else
            real4 posq1 = posq[idx1];
            real4 posq2 = posq[idx2];
            real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#endif
            real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
            real invR = RSQRT(r2);
            real r = r2 * invR;
            real dq = k * (r - b);
#ifdef USE_PBC
            atomicAdd(&posq[indexAtom[idx1]].w, dq);
            atomicAdd(&posq[indexAtom[idx2]].w, -dq);
#else
            atomicAdd(&posq[idx1].w, dq);
            atomicAdd(&posq[idx2].w, -dq);
#endif

            int pair1 = 3 * 4 * npair;
            int pair2 = 3 * (4 * npair + 1);
            int pair3 = 3 * (4 * npair + 2);
            int pair4 = 3 * (4 * npair + 3);
            real constant = k * invR;
            real3 val = constant * delta;
            atomicAdd(&dqdx_val[pair1], -val.x);
            atomicAdd(&dqdx_val[pair2],  val.x);
            atomicAdd(&dqdx_val[pair3],  val.x);
            atomicAdd(&dqdx_val[pair4], -val.x);
            atomicAdd(&dqdx_val[pair1+1], -val.y);
            atomicAdd(&dqdx_val[pair2+1],  val.y);
            atomicAdd(&dqdx_val[pair3+1],  val.y);
            atomicAdd(&dqdx_val[pair4+1], -val.y);
            atomicAdd(&dqdx_val[pair1+2], -val.z);
            atomicAdd(&dqdx_val[pair2+2],  val.z);
            atomicAdd(&dqdx_val[pair3+2],  val.z);
            atomicAdd(&dqdx_val[pair4+2], -val.z);
        } else {
            // angle
            int pidx = npair - NUM_FLUX_BONDS;
            int idx1 = cf_idx[PSHIFT2+pidx*3];
            int idx2 = cf_idx[PSHIFT2+pidx*3+1];
            int idx3 = cf_idx[PSHIFT2+pidx*3+2];
            real k = cf_params[PSHIFT2+pidx*2];
            real theta = cf_params[PSHIFT2+pidx*2+1];
#ifdef USE_PBC
            real4 posq1 = posq[indexAtom[idx1]];
            real4 posq2 = posq[indexAtom[idx2]];
            real4 posq3 = posq[indexAtom[idx3]];
            real3 d21 = make_real3(posq1.x-posq2.x, posq1.y-posq2.y, posq1.z-posq2.z);
            // APPLY_PERIODIC_TO_DELTA(d21)
            real3 d23 = make_real3(posq3.x-posq2.x, posq3.y-posq2.y, posq3.z-posq2.z);
            // APPLY_PERIODIC_TO_DELTA(d23)
            real3 d13 = make_real3(posq3.x-posq1.x, posq3.y-posq1.y, posq3.z-posq1.z);
            // APPLY_PERIODIC_TO_DELTA(d13)
#else
            real4 posq1 = posq[idx1];
            real4 posq2 = posq[idx2];
            real4 posq3 = posq[idx3];
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

            real dq = k * (angle - theta);
#ifdef USE_PBC
            atomicAdd(&posq[indexAtom[idx1]].w, dq);
            atomicAdd(&posq[indexAtom[idx2]].w, dq);
            atomicAdd(&posq[indexAtom[idx3]].w, -2 * dq);
#else
            atomicAdd(&posq[idx1].w, dq);
            atomicAdd(&posq[idx2].w, dq);
            atomicAdd(&posq[idx3].w, -2 * dq);
#endif

            int pair1 = 3 * (PSHIFT4 + 9 * pidx);
            int pair2 = 3 * (PSHIFT4 + 9 * pidx + 1);
            int pair3 = 3 * (PSHIFT4 + 9 * pidx + 2);
            int pair4 = 3 * (PSHIFT4 + 9 * pidx + 3);
            int pair5 = 3 * (PSHIFT4 + 9 * pidx + 4);
            int pair6 = 3 * (PSHIFT4 + 9 * pidx + 5);
            int pair7 = 3 * (PSHIFT4 + 9 * pidx + 6);
            int pair8 = 3 * (PSHIFT4 + 9 * pidx + 7);
            int pair9 = 3 * (PSHIFT4 + 9 * pidx + 8);
            real one_const = k * RSQRT(1 - cost*cost);
            real fin_const1 = invR21 * invR23 * one_const;
            real fin_const2_r21 = cost * one_const * invR21 * invR21;
            real fin_const2_r23 = cost * one_const * invR23 * invR23;

            real3 v1 = - fin_const1 * d23 + fin_const2_r21 * d21;
            real3 v3 = - fin_const1 * d21 + fin_const2_r23 * d23;
            real3 v2 = - v1 - v3;

            atomicAdd(&dqdx_val[pair1], v1.x);
            atomicAdd(&dqdx_val[pair2], v2.x);
            atomicAdd(&dqdx_val[pair3], v3.x);
            atomicAdd(&dqdx_val[pair4], -2 * v1.x);
            atomicAdd(&dqdx_val[pair5], -2 * v2.x);
            atomicAdd(&dqdx_val[pair6], -2 * v3.x);
            atomicAdd(&dqdx_val[pair7], v1.x);
            atomicAdd(&dqdx_val[pair8], v2.x);
            atomicAdd(&dqdx_val[pair9], v3.x);
            atomicAdd(&dqdx_val[pair1+1], v1.y);
            atomicAdd(&dqdx_val[pair2+1], v2.y);
            atomicAdd(&dqdx_val[pair3+1], v3.y);
            atomicAdd(&dqdx_val[pair4+1], -2 * v1.y);
            atomicAdd(&dqdx_val[pair5+1], -2 * v2.y);
            atomicAdd(&dqdx_val[pair6+1], -2 * v3.y);
            atomicAdd(&dqdx_val[pair7+1], v1.y);
            atomicAdd(&dqdx_val[pair8+1], v2.y);
            atomicAdd(&dqdx_val[pair9+1], v3.y);
            atomicAdd(&dqdx_val[pair1+2], v1.z);
            atomicAdd(&dqdx_val[pair2+2], v2.z);
            atomicAdd(&dqdx_val[pair3+2], v3.z);
            atomicAdd(&dqdx_val[pair4+2], -2 * v1.z);
            atomicAdd(&dqdx_val[pair5+2], -2 * v2.z);
            atomicAdd(&dqdx_val[pair6+2], -2 * v3.z);
            atomicAdd(&dqdx_val[pair7+2], v1.z);
            atomicAdd(&dqdx_val[pair8+2], v2.z);
            atomicAdd(&dqdx_val[pair9+2], v3.z);
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
    const real*           __restrict__    dqdx_val
){
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < NUM_DQDX_PAIRS; npair += blockDim.x*gridDim.x){
#ifdef USE_PBC
        int p1 = dqdx_dqidx[npair];
        int p2 = indexAtom[dqdx_dxidx[npair]];
#else
        int p1 = dqdx_dqidx[npair];
        int p2 = dqdx_dxidx[npair];
#endif
        atomicAdd(&forceBuffers[p2], static_cast<unsigned long long>((long long) (-dedq[p1]*dqdx_val[3*npair]*0x100000000)));
        atomicAdd(&forceBuffers[p2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-dedq[p1]*dqdx_val[3*npair+1]*0x100000000)));
        atomicAdd(&forceBuffers[p2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-dedq[p1]*dqdx_val[3*npair+2]*0x100000000)));
    }
}

extern "C" __global__ void printdQdX(
    const int*            __restrict__    dqdx_dqidx,
    const int*            __restrict__    dqdx_dxidx,
    const real*           __restrict__    dqdx_val
){
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < NUM_DQDX_PAIRS; npair += blockDim.x*gridDim.x){
        printf("%i %i %f %f %f\n", dqdx_dqidx[npair], dqdx_dxidx[npair], dqdx_val[3*npair], dqdx_val[3*npair+1], dqdx_val[3*npair+2]);
    }
}