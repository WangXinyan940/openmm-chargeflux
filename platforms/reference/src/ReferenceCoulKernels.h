#ifndef REFERENCE_COUL_KERNELS_H_
#define REFERENCE_COUL_KERNELS_H_

#include "CoulKernels.h"
#include "openmm/Platform.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <vector>
#include <utility>
#include <iostream>

namespace CoulPlugin {

/**
 * This kernel is invoked by CoulForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcCoulForceKernel : public CalcCoulForceKernel {
public:
    ReferenceCalcCoulForceKernel(std::string name, const OpenMM::Platform& platform) : CalcCoulForceKernel(name, platform) {
    }
    ~ReferenceCalcCoulForceKernel();
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
    double cutoff;
    std::vector<double> charges;
    std::vector<std::set<int>> exclusions;
    double ewaldTol, alpha, one_alpha2;
    bool ifPBC;
    int kmaxx, kmaxy, kmaxz;
    OpenMM::NeighborList* neighborList;
};

} // namespace CoulPlugin

#endif /*REFERENCE_COUL_KERNELS_H_*/