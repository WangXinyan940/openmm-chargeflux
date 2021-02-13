#ifndef COUL_KERNELS_H_
#define COUL_KERNELS_H_

#include "CoulForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace CoulPlugin {

/**
 * This kernel is invoked by CoulForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcCoulForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcCoulForce";
    }
    CalcCoulForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the CoulForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const CoulForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

} // namespace CoulPlugin

#endif /*COUL_KERNELS_H_*/