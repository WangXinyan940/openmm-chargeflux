extern "C" __global__ void calcNoPBCEnForces(
    mixed*              __restrict__     energyBuffer,
    const real4*        __restrict__     posq,
    unsigned long long* __restrict__     forceBuffers,
    real*               __restrict__     dedq,
    const real4*        __restrict__     parameters,
    const int*          __restrict__     pairidx0,
    const int*          __restrict__     pairidx1,
    int                                  numParticles,
    int                                  paddedNumAtoms
) {
    int totpair = numParticles * (numParticles - 1) / 2;
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < totpair; npair += blockDim.x*gridDim.x) {
        int ii = pairidx0[npair];
        int jj = pairidx1[npair];
        real3 delta = make_real3(posq[jj].x-posq[ii].x,posq[jj].y-posq[ii].y,posq[jj].z-posq[ii].z);
        real R2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        real inverseR = RSQRT(R2);
        real c1c2 = posq[ii].w * posq[jj].w;

        real sig = parameters[ii].y + parameters[jj].y;
        real sig2 = inverseR * sig;
        sig2 *= sig2;
        real sig6 = sig2 * sig2 * sig2;
        real epssig6 = parameters[ii].z * parameters[jj].z * sig6;

        real ener = ONE_4PI_EPS0 * c1c2 * inverseR + epssig6 * (sig6 - 1);

        atomicAdd(&energyBuffer[ii], ener);
        real dEdRdR = ONE_4PI_EPS0 * c1c2 * inverseR + epssig6 * (12 * sig6 - 6);
        dEdRdR *= inverseR * inverseR;
        real3 force = - dEdRdR * delta;
        atomicAdd(&forceBuffers[ii], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[ii+paddedNumAtoms], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[ii+2*paddedNumAtoms], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
        atomicAdd(&forceBuffers[jj], static_cast<unsigned long long>((long long) (-force.x*0x100000000)));
        atomicAdd(&forceBuffers[jj+paddedNumAtoms], static_cast<unsigned long long>((long long) (-force.y*0x100000000)));
        atomicAdd(&forceBuffers[jj+2*paddedNumAtoms], static_cast<unsigned long long>((long long) (-force.z*0x100000000)));

        atomicAdd(&dedq[ii], ONE_4PI_EPS0*posq[jj].w*inverseR);
        atomicAdd(&dedq[jj], ONE_4PI_EPS0*posq[ii].w*inverseR);
    }
}

extern "C" __global__ void calcNoPBCExclusions(
    mixed*              __restrict__     energyBuffer,
    const real4*        __restrict__     posq,
    unsigned long long* __restrict__     forceBuffers,
    real*               __restrict__     dedq,
    const real4*        __restrict__     parameters,
    const int*          __restrict__     expairidx0,
    const int*          __restrict__     expairidx1,
    const int                            totpair,
    const int                            numParticles,
    const int                            paddedNumAtoms) {
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < totpair; npair += blockDim.x*gridDim.x) {
        int ii = expairidx0[npair];
        int jj = expairidx1[npair];
        real3 delta = make_real3(posq[jj].x-posq[ii].x,posq[jj].y-posq[ii].y,posq[jj].z-posq[ii].z);
        real R2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        real inverseR = RSQRT(R2);
        real c1c2 = posq[ii].w * posq[jj].w;

        real sig = parameters[ii].y + parameters[jj].y;
        real sig2 = inverseR * sig;
        sig2 *= sig2;
        real sig6 = sig2 * sig2 * sig2;
        real epssig6 = parameters[ii].z * parameters[jj].z * sig6;

        real ener = ONE_4PI_EPS0 * c1c2 * inverseR + epssig6 * (sig6 - 1);

        energyBuffer[npair] -= ener;

        real dEdRdR = ONE_4PI_EPS0 * c1c2 * inverseR + epssig6 * (12 * sig6 - 6);
        dEdRdR *= inverseR * inverseR;
        
        real3 force = dEdRdR * delta;
        atomicAdd(&forceBuffers[ii], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[ii+paddedNumAtoms], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[ii+2*paddedNumAtoms], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
        atomicAdd(&forceBuffers[jj], static_cast<unsigned long long>((long long) (-force.x*0x100000000)));
        atomicAdd(&forceBuffers[jj+paddedNumAtoms], static_cast<unsigned long long>((long long) (-force.y*0x100000000)));
        atomicAdd(&forceBuffers[jj+2*paddedNumAtoms], static_cast<unsigned long long>((long long) (-force.z*0x100000000)));

        atomicAdd(&dedq[ii], -ONE_4PI_EPS0*posq[jj].w*inverseR);
        atomicAdd(&dedq[jj], -ONE_4PI_EPS0*posq[ii].w*inverseR);
    }
}