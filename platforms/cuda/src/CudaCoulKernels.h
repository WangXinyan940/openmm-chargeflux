#ifndef CUDA_COUL_KERNELS_H_
#define CUDA_COUL_KERNELS_H_

#include "CoulKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>
#include <string>
#include <utility>

namespace CoulPlugin {

/**
 * This kernel is invoked by CoulForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcCoulForceKernel : public CalcCoulForceKernel {
public:
    CudaCalcCoulForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) :
            CalcCoulForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    }
    ~CudaCalcCoulForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the CoulForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const CoulForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    bool hasInitializedKernel;
    OpenMM::CudaArray charges_cu;
    OpenMM::CudaArray pairidx0, pairidx1;
    OpenMM::CudaArray expairidx0, expairidx1;
    OpenMM::CudaArray cosSinSums;
    OpenMM::CudaArray indexAtom;
    OpenMM::CudaContext& cu;
    CUfunction calcNoPBCEnForcesKernel;
    CUfunction calcNoPBCExclusionsKernel;
    CUfunction calcEwaldRecEnerKernel;
    CUfunction calcEwaldRecForceKernel;
    CUfunction calcEwaldRealKernel;
    CUfunction calcEwaldExclusionsKernel;
    CUfunction indexAtomKernel;
    double cutoff;
    std::vector<std::vector<int>> exclusions;
    int numexclusions;
    double ewaldTol, alpha, one_alpha2;
    double selfEwaldEnergy;
    bool ifPBC;
    int kmaxx, kmaxy, kmaxz;
    double selfenergy;
};

} // namespace CoulPlugin

#endif /*CUDA_COUL_KERNELS_H_*/