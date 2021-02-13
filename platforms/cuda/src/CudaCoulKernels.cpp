#include "CudaCoulKernels.h"
#include "CudaCoulKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaNonbondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/cuda/CudaParameterSet.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "CudaKernelSources.h"
#include <map>
#include <iostream>
#include <set>
#include <utility>
#include <cmath>

using namespace CoulPlugin;
using namespace OpenMM;
using namespace std;

class CudaCalcCoulForceInfo : public CudaForceInfo {
public:
	CudaCalcCoulForceInfo(const CoulForce& force) :
			force(force) {
	}
    bool areParticlesIdentical(int particle1, int particle2) {
        double p1, p2;
        p1 = force.getParticleCharge(particle1);
        p2 = force.getParticleCharge(particle2);
        return (p1 == p2);
    }
	int getNumParticleGroups() {
        int nexp = force.getNumExceptions();
		return nexp;
	}
	void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        force.getExceptionParameters(index, particle1, particle2);
		particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
	}
	bool areGroupsIdentical(int group1, int group2) {
		return true;
	}
private:
	const CoulForce& force;
};

static double getEwaldParamValue(int kmax, double width, double alpha){
    double temp = kmax * M_PI / (width * alpha);
    return 0.05 * sqrt(width * alpha) * kmax * exp(- temp * temp);
}

CudaCalcCoulForceKernel::~CudaCalcCoulForceKernel() {
}

void CudaCalcCoulForceKernel::initialize(const System& system, const CoulForce& force) {
    cu.setAsCurrent();

    int numParticles = system.getNumParticles();
    int elementSize = cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float);

    ifPBC = force.usesPeriodicBoundaryConditions();
    cutoff = force.getCutoffDistance();

    // vector<vector<int>> exclusions;
    exclusions.resize(numParticles);
    for(int ii=0;ii<numParticles;ii++){
        exclusions[ii].push_back(ii);
    }
    // for(int ii=0;ii<force.getNumExceptions();ii++){
    //     int p1, p2;
    //     force.getExceptionParameters(ii, p1, p2);
    //     exclusions[p1].push_back(p2);
    //     exclusions[p2].push_back(p1);
    // }

    // Inititalize CUDA objects.
    // if noPBC
    if (cu.getUseDoublePrecision()){
        vector<double> parameters;
        for(int ii=0;ii<numParticles;ii++){
            double prm = force.getParticleCharge(ii);
            parameters.push_back(prm);
        }
        charges_cu.initialize(cu, numParticles, elementSize, "charges");
        charges_cu.upload(parameters);
    } else {
        vector<float> parameters;
        for(int ii=0;ii<numParticles;ii++){
            float prm = force.getParticleCharge(ii);
            parameters.push_back(prm);
        }
        charges_cu.initialize(cu, numParticles, elementSize, "charges");
        charges_cu.upload(parameters);
    }

    numexclusions = force.getNumExceptions();
    if (numexclusions > 0){
        vector<int> exidx0, exidx1;
        exidx0.resize(force.getNumExceptions());
        exidx1.resize(force.getNumExceptions());
        for(int ii=0;ii<force.getNumExceptions();ii++){
            int p1, p2;
            force.getExceptionParameters(ii, p1, p2);
            exidx0[ii] = p1;
            exidx1[ii] = p2;
        }
        expairidx0.initialize(cu, exidx0.size(), sizeof(int), "exindex0");
        expairidx1.initialize(cu, exidx1.size(), sizeof(int), "exindex1");
        expairidx0.upload(exidx0);
        expairidx1.upload(exidx1);
    }



    if (!ifPBC){
        map<string, string> defines;
        defines["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
        CUmodule module = cu.createModule(CudaKernelSources::vectorOps + CudaCoulKernelSources::noPBCForce, defines);
        calcNoPBCEnForcesKernel = cu.getKernel(module, "calcNoPBCEnForces");
        calcNoPBCExclusionsKernel = cu.getKernel(module, "calcNoPBCExclusions");
        vector<int> idx0;
        vector<int> idx1;
        idx0.resize(numParticles*(numParticles-1)/2);
        idx1.resize(numParticles*(numParticles-1)/2);
        int count = 0;
        for(int ii=0;ii<numParticles;ii++){
            for(int jj=ii+1;jj<numParticles;jj++){
                idx0[count] = ii;
                idx1[count] = jj;
                count += 1;
            }
        }
        pairidx0.initialize(cu, numParticles*(numParticles-1)/2, sizeof(int), "index0");
        pairidx1.initialize(cu, numParticles*(numParticles-1)/2, sizeof(int), "index1");
        pairidx0.upload(idx0);
        pairidx1.upload(idx1);

    } else {

        cu.getNonbondedUtilities().addInteraction(true, true, true, cutoff, exclusions, "", force.getForceGroup());

        set<pair<int, int>> tilesWithExclusions;
        for (int atom1 = 0; atom1 < (int) exclusions.size(); ++atom1) {
            int x = atom1/CudaContext::TileSize;
            for (int atom2 : exclusions[atom1]) {
                int y = atom2/CudaContext::TileSize;
                tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
            }
        }

        vector<int> indexAtomVec;
        indexAtomVec.resize(numParticles);
        indexAtom.initialize(cu, numParticles, sizeof(int), "indexAtom");
        indexAtom.upload(indexAtomVec);

        // alpha - kmax
        ewaldTol = force.getEwaldErrorTolerance();
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        alpha = (1.0/cutoff)*sqrt(-log(2.0*ewaldTol));
        one_alpha2 = 1.0 / alpha / alpha;
        kmaxx = 1;
        while (getEwaldParamValue(kmaxx, boxVectors[0][0], alpha) > ewaldTol){
            kmaxx += 1;
        }
        kmaxy = 1;
        while (getEwaldParamValue(kmaxy, boxVectors[1][1], alpha) > ewaldTol){
            kmaxy += 1;
        }
        kmaxz = 1;
        while (getEwaldParamValue(kmaxz, boxVectors[2][2], alpha) > ewaldTol){
            kmaxz += 1;
        }
        if (kmaxx%2 == 0)
            kmaxx += 1;
        if (kmaxy%2 == 0)
            kmaxy += 1;
        if (kmaxz%2 == 0)
            kmaxz += 1;

        int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        cosSinSums.initialize(cu, 2*(2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1), elementSize, "cosSinSums");

        map<string, string> pbcDefines;
        pbcDefines["NUM_ATOMS"] = cu.intToString(numParticles);
        pbcDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        pbcDefines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        pbcDefines["THREAD_BLOCK_SIZE"] = cu.intToString(cu.getNonbondedUtilities().getForceThreadBlockSize());

        pbcDefines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        int numExclusionTiles = tilesWithExclusions.size();
        pbcDefines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
        int numContexts = cu.getPlatformData().contexts.size();
        int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
        int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
        pbcDefines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
        pbcDefines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);
        pbcDefines["USE_PERIODIC"] = "1";
        pbcDefines["USE_CUTOFF"] = "1";
        pbcDefines["USE_EXCLUSIONS"] = "";
        pbcDefines["USE_SYMMETRIC"] = "1";
        pbcDefines["INCLUDE_FORCES"] = "1";
        pbcDefines["INCLUDE_ENERGY"] = "1";
        pbcDefines["CUTOFF"] = cu.doubleToString(cutoff);

        pbcDefines["USE_DOUBLE_PRECISION"] = cu.getUseDoublePrecision() ? "1" : "";
        pbcDefines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        pbcDefines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        pbcDefines["KMAX_X"] = cu.intToString(kmaxx);
        pbcDefines["KMAX_Y"] = cu.intToString(kmaxy);
        pbcDefines["KMAX_Z"] = cu.intToString(kmaxz);
        pbcDefines["EXP_COEFFICIENT"] = cu.doubleToString(-1.0/(4.0*alpha*alpha));
        pbcDefines["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
        pbcDefines["M_PI"] = cu.doubleToString(M_PI);

        // macro for short-range
        CUmodule PBCModule = cu.createModule(CudaKernelSources::vectorOps + CudaCoulKernelSources::PBCForce, pbcDefines);
        calcEwaldRecEnerKernel = cu.getKernel(PBCModule, "computeEwaldRecEner");
        calcEwaldRecForceKernel = cu.getKernel(PBCModule, "computeEwaldRecForce");
        calcEwaldRealKernel = cu.getKernel(PBCModule, "computeNonbonded");
        calcEwaldExclusionsKernel = cu.getKernel(PBCModule, "computeExclusion");
        indexAtomKernel = cu.getKernel(PBCModule, "genIndexAtom");

        selfEwaldEnergy = 0.0;
        for(int ii=0;ii<numParticles;ii++){
            double chrg = force.getParticleCharge(ii);
            selfEwaldEnergy -= ONE_4PI_EPS0 * chrg * chrg * alpha / sqrt(M_PI);
        }
    }
    hasInitializedKernel = true;

    cu.addForce(new CudaCalcCoulForceInfo(force));
}

double CudaCalcCoulForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    
    int numParticles = cu.getNumAtoms();
    double energy = 0.0;
    if (ifPBC){
        energy += selfEwaldEnergy;
        void* args_rec1[] = {
            &cu.getEnergyBuffer().getDevicePointer(),
            &cu.getPosq().getDevicePointer(),
            &charges_cu.getDevicePointer(),                             // const real*    
            &cu.getAtomIndexArray().getDevicePointer(),             // const int*           
            &cosSinSums.getDevicePointer(),
            cu.getPeriodicBoxSizePointer(),                         // real4                                      periodicBoxSize
            cu.getInvPeriodicBoxSizePointer(),                      // real4    
        };
        cu.executeKernel(calcEwaldRecEnerKernel, args_rec1, (2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1));

        void* args_rec2[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getPosq().getDevicePointer(),
            &charges_cu.getDevicePointer(),                             // const real*    
            &cu.getAtomIndexArray().getDevicePointer(),             // const int*           
            &cosSinSums.getDevicePointer(),
            cu.getPeriodicBoxSizePointer(),                         // real4                                      periodicBoxSize
            cu.getInvPeriodicBoxSizePointer()                       // real4     
        };
        cu.executeKernel(calcEwaldRecForceKernel, args_rec2, numParticles);

        int paddedNumAtoms = cu.getPaddedNumAtoms();
        CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
        int startTileIndex = nb.getStartTileIndex();
        int numTileIndices = nb.getNumTiles();
        unsigned int maxTiles = nb.getInteractingTiles().getSize();
        int maxSinglePairs = nb.getSinglePairs().getSize();

        void* args[] = {
            &cu.getForce().getDevicePointer(),                      // unsigned long long*       __restrict__     forceBuffers, 
            &cu.getEnergyBuffer().getDevicePointer(),               // mixed*                    __restrict__     energyBuffer, 
            &cu.getPosq().getDevicePointer(),                       // const real4*              __restrict__     posq, 
            &charges_cu.getDevicePointer(),                             // const real*               __restrict__     params,
            &cu.getAtomIndexArray().getDevicePointer(),             // const int*                __restrict__     atomIndex,
            &nb.getExclusions().getDevicePointer(),                 // const tileflags*          __restrict__     exclusions,
            &nb.getExclusionTiles().getDevicePointer(),             // const int2*               __restrict__     exclusionTiles,
            &startTileIndex,                                        // unsigned int                               startTileIndex,
            &numTileIndices,                                        // unsigned long long                         numTileIndices,
            &nb.getInteractingTiles().getDevicePointer(),           // const int*                __restrict__     tiles, 
            &nb.getInteractionCount().getDevicePointer(),           // const unsigned int*       __restrict__     interactionCoun
            cu.getPeriodicBoxSizePointer(),                         // real4                                      periodicBoxSize
            cu.getInvPeriodicBoxSizePointer(),                      // real4                                      invPeriodicBoxS
            cu.getPeriodicBoxVecXPointer(),                         // real4                                      periodicBoxVecX
            cu.getPeriodicBoxVecYPointer(),                         // real4                                      periodicBoxVecY
            cu.getPeriodicBoxVecZPointer(),                         // real4                                      periodicBoxVecZ
            &maxTiles,                                              // unsigned int                               maxTiles, 
            &nb.getBlockCenters().getDevicePointer(),               // const real4*              __restrict__     blockCenter,
            &nb.getBlockBoundingBoxes().getDevicePointer(),         // const real4*              __restrict__     blockSize, 
            &nb.getInteractingAtoms().getDevicePointer(),           // const unsigned int*       __restrict__     interactingAtom
            &maxSinglePairs,                                        // unsigned int                               maxSinglePairs,
            &nb.getSinglePairs().getDevicePointer()                // const int2*               __restrict__     singlePairs
        };
        cu.executeKernel(calcEwaldRealKernel, args, nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());

        if (numexclusions > 0){
            void* argSwitch[] = {
                &cu.getAtomIndexArray().getDevicePointer(),
                &indexAtom.getDevicePointer(),
                &numParticles
            };
            cu.executeKernel(indexAtomKernel, argSwitch, numParticles);

            void* argsEx[] = {
                &cu.getForce().getDevicePointer(),            //   forceBuffers, 
                &cu.getEnergyBuffer().getDevicePointer(),     //   energyBuffer, 
                &cu.getPosq().getDevicePointer(),             //   posq, 
                &charges_cu.getDevicePointer(),               //   params,
                &cu.getAtomIndexArray().getDevicePointer(),   //   atomIndex,
                &indexAtom.getDevicePointer(),                //   indexAtom,
                &expairidx0.getDevicePointer(),               //   exclusionidx1,
                &expairidx1.getDevicePointer(),               //   exclusionidx2,
                &numexclusions,                               //   numExclusions,
                cu.getPeriodicBoxSizePointer(),               //   periodicBoxSize, 
                cu.getInvPeriodicBoxSizePointer(),            //   invPeriodicBoxSize, 
                cu.getPeriodicBoxVecXPointer(),               //   periodicBoxVecX, 
                cu.getPeriodicBoxVecYPointer(),               //   periodicBoxVecY, 
                cu.getPeriodicBoxVecZPointer()                //   periodicBoxVecZ
            };
            cu.executeKernel(calcEwaldExclusionsKernel, argsEx, numexclusions);
        }
    } else {
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {
            &cu.getEnergyBuffer().getDevicePointer(), 
            &cu.getPosq().getDevicePointer(), 
            &cu.getForce().getDevicePointer(), 
            &charges_cu.getDevicePointer(), 
            &cu.getAtomIndexArray().getDevicePointer(),
            &pairidx0.getDevicePointer(), 
            &pairidx1.getDevicePointer(), 
            &numParticles, &paddedNumAtoms
        };
        cu.executeKernel(calcNoPBCEnForcesKernel, args, numParticles*(numParticles-1)/2);

        if (numexclusions > 0){
            void* args2[] = {
                &cu.getEnergyBuffer().getDevicePointer(), 
                &cu.getPosq().getDevicePointer(), 
                &cu.getForce().getDevicePointer(), 
                &charges_cu.getDevicePointer(), 
                &cu.getAtomIndexArray().getDevicePointer(),
                &expairidx0.getDevicePointer(), 
                &expairidx1.getDevicePointer(), 
                &numexclusions, 
                &numParticles, 
                &paddedNumAtoms
            };
            cu.executeKernel(calcNoPBCExclusionsKernel, args2, numexclusions);
        }
    }
    return energy;
}