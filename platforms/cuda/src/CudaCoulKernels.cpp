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
        vector<double> realc;
        for(int ii=0;ii<numParticles;ii++){
            double prm = force.getParticleCharge(ii);
            parameters.push_back(prm);
            realc.push_back(prm);
        }
        charges_cu.initialize(cu, numParticles, elementSize, "charges");
        charges_cu.upload(parameters);
        realcharges_cu.initialize(cu, numParticles, elementSize, "realcharges");
        realcharges_cu.upload(realc);

        vector<int> fbidx, faidx;
        vector<double> fbprms, faprms;
        numFluxBonds = force.getNumFluxBonds();
        numFluxAngles = force.getNumFluxAngles();
        if (numFluxBonds > 0){
            for(int ii=0;ii<numFluxBonds;ii++){
                int idx1, idx2;
                double k, b;
                force.getFluxBondParameters(ii, idx1, idx2, k, b);
                fbidx.push_back(idx1);
                fbidx.push_back(idx2);
                fbprms.push_back(k);
                fbprms.push_back(b);
            }
            fbond_idx.initialize(cu, numFluxBonds*2, sizeof(int), "fbondidx");
            fbond_idx.upload(fbidx);
            fbond_params.initialize(cu, numFluxBonds*2, elementSize, "fbondprms");
            fbond_params.upload(fbprms);
        }

        if (numFluxAngles > 0){
            for(int ii=0;ii<numFluxAngles;ii++){
                int idx1, idx2, idx3;
                double k, theta;
                force.getFluxAngleParameters(ii, idx1, idx2, idx3, k, theta);
                faidx.push_back(idx1);
                faidx.push_back(idx2);
                faidx.push_back(idx3);
                faprms.push_back(k);
                faprms.push_back(theta);
            }
            fangle_idx.initialize(cu, numFluxAngles*3, sizeof(int), "fangleidx");
            fangle_idx.upload(faidx);
            fangle_params.initialize(cu, numFluxAngles*2, elementSize, "fangleprms");
            fangle_params.upload(faprms);
        }

    } else {
        vector<float> parameters;
        vector<float> realc;
        for(int ii=0;ii<numParticles;ii++){
            float prm = force.getParticleCharge(ii);
            parameters.push_back(prm);
            realc.push_back(prm);
        }
        charges_cu.initialize(cu, numParticles, elementSize, "charges");
        charges_cu.upload(parameters);
        realcharges_cu.initialize(cu, numParticles, elementSize, "realcharges");
        realcharges_cu.upload(realc);

        vector<int> fbidx, faidx;
        vector<float> fbprms, faprms;
        numFluxBonds = force.getNumFluxBonds();
        numFluxAngles = force.getNumFluxAngles();

        if (numFluxBonds > 0){
            for(int ii=0;ii<numFluxBonds;ii++){
                int idx1, idx2;
                double k, b;
                force.getFluxBondParameters(ii, idx1, idx2, k, b);
                fbidx.push_back(idx1);
                fbidx.push_back(idx2);
                fbprms.push_back(k);
                fbprms.push_back(b);
            }
            fbond_idx.initialize(cu, numFluxBonds*2, sizeof(int), "fbondidx");
            fbond_idx.upload(fbidx);
            fbond_params.initialize(cu, numFluxBonds*2, elementSize, "fbondprms");
            fbond_params.upload(fbprms);
        }

        if (numFluxAngles > 0){
            for(int ii=0;ii<numFluxAngles;ii++){
                int idx1, idx2, idx3;
                double k, theta;
                force.getFluxAngleParameters(ii, idx1, idx2, idx3, k, theta);
                faidx.push_back(idx1);
                faidx.push_back(idx2);
                faidx.push_back(idx3);
                faprms.push_back(k);
                faprms.push_back(theta);
            }
            fangle_idx.initialize(cu, numFluxAngles*3, sizeof(int), "fangleidx");
            fangle_idx.upload(faidx);
            fangle_params.initialize(cu, numFluxAngles*2, elementSize, "fangleprms");
            fangle_params.upload(faprms);
        }
    }

    if (numFluxAngles + numFluxBonds > 0){
        vector<int> dqdx_dqidx_v;
        vector<int> dqdx_dxidx_v;

        for(int ii=0;ii<numFluxBonds;ii++){
            int p1, p2;
            double k, b;
            force.getFluxBondParameters(ii, p1, p2, k, b);
            // p1-p1
            dqdx_dqidx_v.push_back(p1);
            dqdx_dxidx_v.push_back(p1);

            // p1-p2
            dqdx_dqidx_v.push_back(p1);
            dqdx_dxidx_v.push_back(p2);

            // p2-p1
            dqdx_dqidx_v.push_back(p2);
            dqdx_dxidx_v.push_back(p1);

            // p2-p2
            dqdx_dqidx_v.push_back(p2);
            dqdx_dxidx_v.push_back(p2);

        }

        for(int ii=0;ii<numFluxAngles;ii++){
            int p1, p2, p3;
            double k, theta;
            force.getFluxAngleParameters(ii, p1, p2, p3, k, theta);
            // p1-p1
            dqdx_dqidx_v.push_back(p1);
            dqdx_dxidx_v.push_back(p1);
            // p1-p2
            dqdx_dqidx_v.push_back(p1);
            dqdx_dxidx_v.push_back(p2);
            // p1-p3
            dqdx_dqidx_v.push_back(p1);
            dqdx_dxidx_v.push_back(p3);
            // p2-p1
            dqdx_dqidx_v.push_back(p2);
            dqdx_dxidx_v.push_back(p1);
            // p2-p2
            dqdx_dqidx_v.push_back(p2);
            dqdx_dxidx_v.push_back(p2);
            // p2-p3
            dqdx_dqidx_v.push_back(p2);
            dqdx_dxidx_v.push_back(p3);
            // p3-p1
            dqdx_dqidx_v.push_back(p3);
            dqdx_dxidx_v.push_back(p1);
            // p3-p2
            dqdx_dqidx_v.push_back(p3);
            dqdx_dxidx_v.push_back(p2);
            // p3-p3
            dqdx_dqidx_v.push_back(p3);
            dqdx_dxidx_v.push_back(p3);
        }

        dqdx_dqidx.initialize(cu, dqdx_dqidx_v.size(), sizeof(int), "dqdx_dqidx");
        dqdx_dqidx.upload(dqdx_dqidx_v);
        dqdx_dxidx.initialize(cu, dqdx_dxidx_v.size(), sizeof(int), "dqdx_dxidx");
        dqdx_dxidx.upload(dqdx_dxidx_v);

        if (cu.getUseDoublePrecision()){

            vector<double> dqdx_val_v;

            for(int ii=0;ii<dqdx_dqidx_v.size()*3;ii++){
                dqdx_val_v.push_back(0);
            }

            dqdx_val.initialize(cu, dqdx_val_v.size(), elementSize, "dqdx_val");
            dqdx_val.upload(dqdx_val_v);
        } else {

            vector<float> dqdx_val_v;

            for(int ii=0;ii<dqdx_dqidx_v.size()*3;ii++){
                dqdx_val_v.push_back(0);
            }

            dqdx_val.initialize(cu, dqdx_val_v.size(), elementSize, "dqdx_val");
            dqdx_val.upload(dqdx_val_v);
        }
    }
    if (cu.getUseDoublePrecision()){
        vector<double> dedq_v;
        for(int ii=0;ii<numParticles;ii++){
            dedq_v.push_back(0);
        }
        dedq.initialize(cu, dedq_v.size(), elementSize, "dedq");
        dedq.upload(dedq_v);
    } else {
        vector<float> dedq_v;
        for(int ii=0;ii<numParticles;ii++){
            dedq_v.push_back(0);
        }
        dedq.initialize(cu, dedq_v.size(), elementSize, "dedq");
        dedq.upload(dedq_v);
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

    map<string, string> defRealCharges;
    if (ifPBC){
        defRealCharges["USE_PBC"] = "1";
    }
    defRealCharges["NUM_FLUX_BONDS"] = cu.intToString(numFluxBonds);
    defRealCharges["NUM_FLUX_ANGLES"] = cu.intToString(numFluxAngles);
    defRealCharges["NUM_ATOMS"] = cu.intToString(numParticles);
    defRealCharges["NUM_DQDX_PAIRS"] = cu.intToString(numFluxBonds * 4 + numFluxAngles * 9);
    defRealCharges["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());

    CUmodule module = cu.createModule(CudaKernelSources::vectorOps + CudaCoulKernelSources::calcChargeFlux, defRealCharges);
    calcRealChargeKernel = cu.getKernel(module, "calcRealCharge");
    copyChargeKernel = cu.getKernel(module, "copyCharge");
    multdQdXKernel = cu.getKernel(module, "multdQdX");
    printdQdXKernel = cu.getKernel(module, "printdQdX");

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
        pbcDefines["ONE_OVER_SQRT_PI"] = cu.doubleToString(1.0/sqrt(M_PI));
        pbcDefines["KMAX_X"] = cu.intToString(kmaxx);
        pbcDefines["KMAX_Y"] = cu.intToString(kmaxy);
        pbcDefines["KMAX_Z"] = cu.intToString(kmaxz);
        pbcDefines["EXP_COEFFICIENT"] = cu.doubleToString(-1.0/(4.0*alpha*alpha));
        pbcDefines["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
        pbcDefines["M_PI"] = cu.doubleToString(M_PI);

        // macro for short-range
        CUmodule PBCModule = cu.createModule(CudaKernelSources::vectorOps + CudaCoulKernelSources::PBCForce, pbcDefines);
        calcEwaldSelfEnerKernel = cu.getKernel(PBCModule, "computeEwaldSelfEner");
        calcEwaldRecEnerKernel = cu.getKernel(PBCModule, "computeEwaldRecEner");
        calcEwaldRecForceKernel = cu.getKernel(PBCModule, "computeEwaldRecForce");
        calcEwaldRealKernel = cu.getKernel(PBCModule, "computeNonbonded");
        calcEwaldExclusionsKernel = cu.getKernel(PBCModule, "computeExclusion");
        indexAtomKernel = cu.getKernel(PBCModule, "genIndexAtom");
    }
    hasInitializedKernel = true;

    cu.addForce(new CudaCalcCoulForceInfo(force));
}

double CudaCalcCoulForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    int numParticles = cu.getNumAtoms();
    double energy = 0.0;


    if (ifPBC){
        void* argSwitch[] = {
            &cu.getAtomIndexArray().getDevicePointer(),
            &indexAtom.getDevicePointer(),
            &numParticles
        };
        cu.executeKernel(indexAtomKernel, argSwitch, numParticles);

        void* argUpdateCharge[] = {
            &realcharges_cu.getDevicePointer(),
            &dedq.getDevicePointer(),
            &charges_cu.getDevicePointer()
        };
        cu.executeKernel(copyChargeKernel, argUpdateCharge, numParticles);

        if (numFluxAngles + numFluxBonds > 0){
            void* args_realc[] = {
                &realcharges_cu.getDevicePointer(),
                &dqdx_val.getDevicePointer(),
                &cu.getPosq().getDevicePointer(),
                &fbond_idx.getDevicePointer(),
                &fbond_params.getDevicePointer(),
                &fangle_idx.getDevicePointer(),
                &fangle_params.getDevicePointer(),
                &indexAtom.getDevicePointer(),
                cu.getPeriodicBoxSizePointer(),      
                cu.getInvPeriodicBoxSizePointer(),   
                cu.getPeriodicBoxVecXPointer(),      
                cu.getPeriodicBoxVecYPointer(),      
                cu.getPeriodicBoxVecZPointer()       
            };
            cu.executeKernel(calcRealChargeKernel, args_realc, numFluxBonds + numFluxAngles);
        }
        void* args_self[] = {
            &cu.getEnergyBuffer().getDevicePointer(),
            &dedq.getDevicePointer(),
            &realcharges_cu.getDevicePointer()
        };
        cu.executeKernel(calcEwaldSelfEnerKernel, args_self, numParticles);
        void* args_rec1[] = {
            &cu.getEnergyBuffer().getDevicePointer(),
            &cu.getPosq().getDevicePointer(),
            &realcharges_cu.getDevicePointer(),                             // const real*    
            &cu.getAtomIndexArray().getDevicePointer(),             // const int*           
            &cosSinSums.getDevicePointer(),
            cu.getPeriodicBoxSizePointer(),                         // real4                                      periodicBoxSize
            cu.getInvPeriodicBoxSizePointer(),                      // real4    
        };
        // cu.executeKernel(calcEwaldRecEnerKernel, args_rec1, (2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1));

        void* args_rec2[] = {
            &cu.getForce().getDevicePointer(),
            &dedq.getDevicePointer(),
            &cu.getPosq().getDevicePointer(),
            &realcharges_cu.getDevicePointer(),                             // const real*    
            &cu.getAtomIndexArray().getDevicePointer(),             // const int*           
            &cosSinSums.getDevicePointer(),
            cu.getPeriodicBoxSizePointer(),                         // real4                                      periodicBoxSize
            cu.getInvPeriodicBoxSizePointer()                       // real4     
        };
        // cu.executeKernel(calcEwaldRecForceKernel, args_rec2, numParticles);

        int paddedNumAtoms = cu.getPaddedNumAtoms();
        CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
        int startTileIndex = nb.getStartTileIndex();
        int numTileIndices = nb.getNumTiles();
        unsigned int maxTiles = nb.getInteractingTiles().getSize();
        int maxSinglePairs = nb.getSinglePairs().getSize();

        void* args[] = {
            &cu.getForce().getDevicePointer(),                      // unsigned long long*       __restrict__     forceBuffers, 
            &cu.getEnergyBuffer().getDevicePointer(),               // mixed*                    __restrict__     energyBuffer, 
            &dedq.getDevicePointer(),
            &cu.getPosq().getDevicePointer(),                       // const real4*              __restrict__     posq, 
            &realcharges_cu.getDevicePointer(),                             // const real*               __restrict__     params,
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
        // cu.executeKernel(calcEwaldRealKernel, args, nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());


        if (numexclusions > 0){
            void* argsEx[] = {
                &cu.getForce().getDevicePointer(),            //   forceBuffers, 
                &cu.getEnergyBuffer().getDevicePointer(),     //   energyBuffer, 
                &dedq.getDevicePointer(),
                &cu.getPosq().getDevicePointer(),             //   posq, 
                &realcharges_cu.getDevicePointer(),               //   params,
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
        if (numFluxAngles + numFluxBonds > 0) {
            void* argsMult[] = {
                &cu.getForce().getDevicePointer(),    // unsigned long long*   __restrict__    forceBuffers, 
                &dedq.getDevicePointer(),             // const real*           __restrict__    dedq,
                &dqdx_dqidx.getDevicePointer(),       // const int*            __restrict__    dqdx_dqidx,
                &dqdx_dxidx.getDevicePointer(),       // const int*            __restrict__    dqdx_dxidx,
                &dqdx_val.getDevicePointer()          // const real*           __restrict__    dqdx_val
            };
            cu.executeKernel(multdQdXKernel, argsMult, 4*numFluxBonds+9*numFluxAngles);
        }
        // if (numFluxAngles + numFluxBonds > 0){
        //     void* argsPrint[] = {
        //         &dqdx_dqidx.getDevicePointer(),       // const int*            __restrict__    dqdx_dqidx,
        //         &dqdx_dxidx.getDevicePointer(),       // const int*            __restrict__    dqdx_dxidx,
        //         &dqdx_val.getDevicePointer()          // const real*           __restrict__    dqdx_val
        //     };
        //     cu.executeKernel(printdQdXKernel, argsPrint, 4*numFluxBonds+9*numFluxAngles);
        // }
    } else {
        void* argUpdateCharge[] = {
            &realcharges_cu.getDevicePointer(),
            &dedq.getDevicePointer(),
            &charges_cu.getDevicePointer()
        };
        cu.executeKernel(copyChargeKernel, argUpdateCharge, numParticles);
        if (numFluxAngles + numFluxBonds > 0){
            void* args_realc[] = {
                &realcharges_cu.getDevicePointer(),
                &dqdx_val.getDevicePointer(),
                &cu.getPosq().getDevicePointer(),
                &fbond_idx.getDevicePointer(),
                &fbond_params.getDevicePointer(),
                &fangle_idx.getDevicePointer(),
                &fangle_params.getDevicePointer()
            };
            cu.executeKernel(calcRealChargeKernel, args_realc, numFluxBonds + numFluxAngles);
        }

        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {
            &cu.getEnergyBuffer().getDevicePointer(), 
            &cu.getPosq().getDevicePointer(), 
            &cu.getForce().getDevicePointer(), 
            &realcharges_cu.getDevicePointer(), 
            &dedq.getDevicePointer(),
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
                &realcharges_cu.getDevicePointer(), 
                &dedq.getDevicePointer(),
                &cu.getAtomIndexArray().getDevicePointer(),
                &expairidx0.getDevicePointer(), 
                &expairidx1.getDevicePointer(), 
                &numexclusions, 
                &numParticles, 
                &paddedNumAtoms
            };
            cu.executeKernel(calcNoPBCExclusionsKernel, args2, numexclusions);
        }
        if (numFluxAngles + numFluxBonds > 0) {
            void* argsMult[] = {
                &cu.getForce().getDevicePointer(),    // unsigned long long*   __restrict__    forceBuffers, 
                &dedq.getDevicePointer(),             // const real*           __restrict__    dedq,
                &dqdx_dqidx.getDevicePointer(),       // const int*            __restrict__    dqdx_dqidx,
                &dqdx_dxidx.getDevicePointer(),       // const int*            __restrict__    dqdx_dxidx,
                &dqdx_val.getDevicePointer()          // const real*           __restrict__    dqdx_val
            };
            cu.executeKernel(multdQdXKernel, argsMult, 4*numFluxBonds+9*numFluxAngles);
        }
        // if (numFluxAngles + numFluxBonds > 0){
        //     void* argsPrint[] = {
        //         &dqdx_dqidx.getDevicePointer(),       // const int*            __restrict__    dqdx_dqidx,
        //         &dqdx_dxidx.getDevicePointer(),       // const int*            __restrict__    dqdx_dxidx,
        //         &dqdx_val.getDevicePointer()          // const real*           __restrict__    dqdx_val
        //     };
        //     cu.executeKernel(printdQdXKernel, argsPrint, 4*numFluxBonds+9*numFluxAngles);
        // }
    }
    return energy;
}