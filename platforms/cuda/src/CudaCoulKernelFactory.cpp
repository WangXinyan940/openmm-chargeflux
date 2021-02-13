#include <exception>

#include "CudaCoulKernelFactory.h"
#include "CudaCoulKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace CoulPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        int argc = 0;
        vector<char**> argv = {NULL};
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaCoulKernelFactory* factory = new CudaCoulKernelFactory();
        platform.registerKernelFactory(CalcCoulForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerCoulCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaCoulKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcCoulForceKernel::Name())
        return new CudaCalcCoulForceKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}