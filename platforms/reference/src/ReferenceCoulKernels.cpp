#include "ReferenceCoulKernels.h"
#include "CoulForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/ReferenceForce.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <cmath>

using namespace OpenMM;
using namespace std;
using namespace CoulPlugin;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

ReferenceCalcCoulForceKernel::~ReferenceCalcCoulForceKernel() {
}

static double getEwaldParamValue(int kmax, double width, double alpha){
    double temp = kmax * M_PI / (width * alpha);
    return 0.05 * sqrt(width * alpha) * kmax * exp(- temp * temp);
}

void ReferenceCalcCoulForceKernel::initialize(const System& system, const CoulForce& force) {
    int numParticles = system.getNumParticles();
    charges.resize(numParticles);
    for(int i=0;i<numParticles;i++){
        charges[i] = force.getParticleCharge(i);
    }
    exclusions.resize(numParticles);
    for(int ii=0;ii<force.getNumExceptions();ii++){
        int p1, p2;
        force.getExceptionParameters(ii, p1, p2);
        exclusions[p1].insert(p2);
        exclusions[p2].insert(p1);
    }

    ifPBC = force.usesPeriodicBoundaryConditions();
    if (ifPBC){
        neighborList = new NeighborList();
        cutoff = force.getCutoffDistance();
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
    }
}

double ReferenceCalcCoulForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& forces = extractForces(context);
    Vec3* box = extractBoxVectors(context);
    int numParticles = pos.size();
    double energy = 0.0;    
    double dEdR;
    vector<double> deltaR;
    deltaR.resize(5);
    if (!ifPBC){
        // noPBC, calcall
        for(int ii=0;ii<numParticles;ii++){
            for(int jj=ii+1;jj<numParticles;jj++){
                ReferenceForce::getDeltaR(pos[ii], pos[jj], &deltaR[0]);
                double inverseR = 1.0 / deltaR[4];
                if (includeEnergy) {
                    energy += ONE_4PI_EPS0*charges[ii]*charges[jj]*inverseR;
                }
                if (includeForces) {
                    dEdR = ONE_4PI_EPS0*charges[ii]*charges[jj]*inverseR*inverseR*inverseR;
                    for(int dd=0;dd<3;dd++){
                        forces[ii][dd] -= dEdR*deltaR[dd];
                        forces[jj][dd] += dEdR*deltaR[dd];
                    }
                }
            }
        }
        // calc exclusions
        for(int p1=0;p1<numParticles;p1++){
            for(set<int>::iterator iter=exclusions[p1].begin(); iter != exclusions[p1].end(); iter++){
                int p2 = *iter;
                if (p1 < p2) {
                    ReferenceForce::getDeltaR(pos[p1], pos[p2], &deltaR[0]);
                    double inverseR = 1.0 / deltaR[4];
                    if (includeEnergy) {
                        energy -= ONE_4PI_EPS0*charges[p1]*charges[p2]*inverseR;
                    }
                    if (includeForces) {
                        dEdR = ONE_4PI_EPS0*charges[p1]*charges[p2]*inverseR*inverseR*inverseR;
                        for(int dd=0;dd<3;dd++){
                            forces[p1][dd] += dEdR*deltaR[dd];
                            forces[p2][dd] -= dEdR*deltaR[dd];
                        }
                    }
                }
            }
        }
    } else {
        // PBC
        // calc self energy
        double selfEwaldEnergy = 0.0;
        double recipEnergy = 0.0;
        double realSpaceEwaldEnergy = 0.0;
        double realSpaceException = 0.0;
        for(int ii=0;ii<numParticles;ii++){
            selfEwaldEnergy -= ONE_4PI_EPS0 * charges[ii] * charges[ii] * alpha / sqrt(M_PI);
        }
        // calc reciprocal part
        
        double recipX = 2 * M_PI / box[0][0];
        double recipY = 2 * M_PI / box[1][1];
        double recipZ = 2 * M_PI / box[2][2];

        double constant = 4.0 / box[0][0] / box[1][1] / box[2][2] * M_PI * ONE_4PI_EPS0;
        
        int minky = 0;
        int minkz = 1;
        for(int nkx=0;nkx < kmaxx;nkx++){
            double kx = nkx * recipX;
            for(int nky=minky;nky < kmaxy;nky++){
                double ky = nky * recipY;
                for(int nkz=minkz;nkz < kmaxz;nkz++){
                    double kz = nkz * recipZ;
                    double k2 = kx * kx + ky * ky + kz * kz;
                    double eak = exp(- k2 * 0.25 * one_alpha2) / k2;
                    double ss = 0.0;
                    double cs = 0.0;
                    if (includeForces || includeEnergy){
                        for(int ii=0;ii<numParticles;ii++){
                            double gr = kx * pos[ii][0] + ky * pos[ii][1] + kz * pos[ii][2];
                            cs += charges[ii] * cos(gr);
                            ss += charges[ii] * sin(gr);
                        }
                    }
                    if(includeForces){
                        for(int ii=0;ii<numParticles;ii++){
                            double gr = kx * pos[ii][0] + ky * pos[ii][1] + kz * pos[ii][2];
                            double gradr = 2.0 * constant * eak * (ss * charges[ii] * cos(gr) - cs * charges[ii] * sin(gr));
                            forces[ii][0] -= gradr * kx;
                            forces[ii][1] -= gradr * ky;
                            forces[ii][2] -= gradr * kz;
                        }
                    }
                    if(includeEnergy){
                        recipEnergy += constant * eak * (cs * cs + ss * ss);
                    }
                }
                minkz = 1 - kmaxz;
            }
            minky = 1 - kmaxy;
        }
        // calc bonded part

        computeNeighborListVoxelHash(*neighborList, numParticles, pos, exclusions, box, ifPBC, cutoff, 0.0);
        
        for(auto& pair : *neighborList){
            int ii = pair.first;
            int jj = pair.second;

            double deltaR[2][ReferenceForce::LastDeltaRIndex];
            ReferenceForce::getDeltaRPeriodic(pos[jj], pos[ii], box, deltaR[0]);
            double r         = deltaR[0][ReferenceForce::RIndex];
            double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
            double alphaR = alpha * r;

            if(includeForces){
                double dEdR = ONE_4PI_EPS0 * charges[ii] * charges[jj] * inverseR * inverseR * inverseR;
                dEdR = dEdR * (erfc(alphaR) + alphaR * exp (- alphaR * alphaR) * 2.0 / sqrt(M_PI));
                for(int kk=0;kk<3;kk++){
                    double fconst = dEdR*deltaR[0][kk];
                    forces[ii][kk] += fconst;
                    forces[jj][kk] -= fconst;
                }
            }

            realSpaceEwaldEnergy += ONE_4PI_EPS0*charges[ii]*charges[jj]*inverseR*erfc(alphaR);
        }

        for(int p1=0;p1<numParticles;p1++){
            for(set<int>::iterator iter=exclusions[p1].begin(); iter != exclusions[p1].end(); iter++){
                int p2 = *iter;
                if (p1 < p2) {
                    double deltaR[2][ReferenceForce::LastDeltaRIndex];
                    ReferenceForce::getDeltaRPeriodic(pos[p2], pos[p1], box, deltaR[0]);
                    // ReferenceForce::getDeltaRPeriodic(pos[p2], pos[p1], deltaR[0]);
                    double r         = deltaR[0][ReferenceForce::RIndex];
                    double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
                    double alphaR = alpha * r;

                    if(includeForces){
                        double dEdR = ONE_4PI_EPS0 * charges[p1] * charges[p2] * inverseR * inverseR * inverseR;
                        dEdR = dEdR * (erf(alphaR) - alphaR * exp (- alphaR * alphaR) * 2.0 / sqrt(M_PI));
                        for(int kk=0;kk<3;kk++){
                            double fconst = dEdR*deltaR[0][kk];
                            forces[p1][kk] -= fconst;
                            forces[p2][kk] += fconst;
                        }
                    }

                    realSpaceException -= ONE_4PI_EPS0*charges[p1]*charges[p2]*inverseR*erf(alphaR);
                }
            }
        }

        energy = selfEwaldEnergy + recipEnergy + realSpaceEwaldEnergy + realSpaceException;
    } 
    return energy;
}