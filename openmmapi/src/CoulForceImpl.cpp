#include "internal/CoulForceImpl.h"
#include "CoulKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace CoulPlugin;
using namespace OpenMM;
using namespace std;

CoulForceImpl::CoulForceImpl(const CoulForce& owner) : owner(owner) {
}

CoulForceImpl::~CoulForceImpl() {
}

void CoulForceImpl::initialize(ContextImpl& context) {

    // Create the kernel.
    kernel = context.getPlatform().createKernel(CalcCoulForceKernel::Name(), context);
    kernel.getAs<CalcCoulForceKernel>().initialize(context.getSystem(), owner);
}

double CoulForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcCoulForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> CoulForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcCoulForceKernel::Name());
    return names;
}