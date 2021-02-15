extern "C" __global__ void copyCharge(
    real*            __restrict__  realcharges,
    const real*      __restrict__  charges
){
    for (int natom = blockIdx.x*blockDim.x+threadIdx.x; natom < NUM_ATOMS; natom += blockDim.x*gridDim.x){
        real newc = charges[natom];
        realcharges[natom] = newc;
    }
}

extern "C" __global__ void calcRealCharge(
    real*             __restrict__  realcharges,
    const real4*      __restrict__  posq,
#ifdef USE_PBC
    const int*        __restrict__  fbond_idx,
    const real*       __restrict__  fbond_params,
    const int*        __restrict__  fangle_idx,
    const real*       __restrict__  fangle_params,
    const int*        __restrict__  indexAtom,
    real4                           periodicBoxSize, 
    real4                           invPeriodicBoxSize, 
    real4                           periodicBoxVecX, 
    real4                           periodicBoxVecY, 
    real4                           periodicBoxVecZ
#else
    const int*        __restrict__  fbond_idx,
    const real*       __restrict__  fbond_params,
    const int*        __restrict__  fangle_idx,
    const real*       __restrict__  fangle_params
#endif
){
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < NUM_FLUX_BONDS + NUM_FLUX_ANGLES; npair += blockDim.x*gridDim.x){
        if (npair < NUM_FLUX_BONDS){
            // bond
            int idx1 = fbond_idx[npair*2];
            int idx2 = fbond_idx[npair*2+1];
            real k = fbond_params[npair*2];
            real b = fbond_params[npair*2+1];
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
            atomicAdd(&realcharges[idx1], dq);
            atomicAdd(&realcharges[idx2], -dq);
        } else {
            // angle
            int pidx = npair - NUM_FLUX_BONDS;
            int idx1 = fangle_idx[pidx*3];
            int idx2 = fangle_idx[pidx*3+1];
            int idx3 = fangle_idx[pidx*3+2];
            real k = fangle_params[pidx*2];
            real theta = fangle_params[pidx*2+1];
#ifdef USE_PBC
            real4 posq1 = posq[indexAtom[idx1]];
            real4 posq2 = posq[indexAtom[idx2]];
            real4 posq3 = posq[indexAtom[idx3]];
            real3 d21 = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
            APPLY_PERIODIC_TO_DELTA(d21)
            real3 d23 = make_real3(posq2.x-posq3.x, posq2.y-posq3.y, posq2.z-posq3.z);
            APPLY_PERIODIC_TO_DELTA(d23)
            real3 d13 = make_real3(posq1.x-posq3.x, posq1.y-posq3.y, posq1.z-posq3.z);
            APPLY_PERIODIC_TO_DELTA(d13)
#else
            real4 posq1 = posq[idx1];
            real4 posq2 = posq[idx2];
            real4 posq3 = posq[idx3];
            real3 d21 = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
            real3 d23 = make_real3(posq2.x-posq3.x, posq2.y-posq3.y, posq2.z-posq3.z);
            real3 d13 = make_real3(posq1.x-posq3.x, posq1.y-posq3.y, posq1.z-posq3.z);
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

            real angle = ACOS((r23_2 + r21_2 - r13_2) * 0.5 * invR21 * invR23);

            real dq = k * (angle - theta);
            atomicAdd(&realcharges[idx1], dq);
            atomicAdd(&realcharges[idx3], dq);
            atomicAdd(&realcharges[idx2], -2 * dq);
        }
    }
}