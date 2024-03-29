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

    void updateRealCharge(std::vector<OpenMM::Vec3>& pos, OpenMM::Vec3* box);
private:
    double cutoff;
    std::vector<double> charges;
    std::vector<double> ljparams;
    std::vector<std::set<int>> exclusions;
    double ewaldTol, alpha, one_alpha2;
    bool ifPBC;
    int kmaxx, kmaxy, kmaxz;
    std::vector<double> realcharges;
    int numCFBonds;
    std::vector<int> fbond_idx;
    std::vector<double> fbond_params;
    int numCFAngles;
    std::vector<int> fangle_idx;
    std::vector<double> fangle_params;
    int numCFWaters;
    std::vector<int> fwater_idx;
    std::vector<double> fwater_params;
    std::vector<int> dqdx_dqidx;
    std::vector<int> dqdx_dxidx; 
    std::vector<double> dqdx_val;

    OpenMM::NeighborList* neighborList;
};

} // namespace CoulPlugin

#endif /*REFERENCE_COUL_KERNELS_H_*/