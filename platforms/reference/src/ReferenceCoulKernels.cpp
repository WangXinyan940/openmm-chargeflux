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

void ReferenceCalcCoulForceKernel::updateRealCharge(vector<Vec3>& pos, Vec3* box){
    for(int ii=0;ii<charges.size();ii++){
        realcharges[ii] = charges[ii];
    }
    vector<double> deltaR1, deltaR2;
    for(int ii=0;ii<numCFBonds;ii++){
        // dc1 = k * (v21 - b)
        // dc2 = - k * (v21 - b)
        int p1 = fbond_idx[2*ii]; // O
        int p2 = fbond_idx[2*ii+1]; // H
        double k = fbond_params[2*ii];
        double b = fbond_params[2*ii+1];
        // calc r
        // delta p1 -> p2
        Vec3 delta;
        if (!ifPBC){
            delta = ReferenceForce::getDeltaR(pos[p1], pos[p2]);
        } else {
            delta = ReferenceForce::getDeltaRPeriodic(pos[p1], pos[p2], box);
        }
        double r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
        double r = sqrt(r2);
        // do something for dq
        double dq = k * (r - b);
        realcharges[p1] += dq;
        realcharges[p2] -= dq;

        // do something for dqdx
        double constant = k / r;
        // p1/p1
        // pair: 4*ii
        // val: 3*pair, 3*pair+1, 3*pair+2
        int pair1 = 4 * ii;
        int pair2 = 4 * ii + 1;
        int pair3 = 4 * ii + 2;
        int pair4 = 4 * ii + 3;
        for(int jj=0;jj<3;jj++){
            double val = constant * delta[jj];
            dqdx_val[3*pair1+jj] = - val;
            dqdx_val[3*pair2+jj] = val;
            dqdx_val[3*pair3+jj] = val;
            dqdx_val[3*pair4+jj] = - val;
        }
    }
    for(int ii=0;ii<numCFAngles;ii++){
        // angle 1-2-3
        // dc1 = k * (a123 - theta)
        // dc3 = k * (a123 - theta)
        // dc2 = - dc1 - dc3
        int p1 = fangle_idx[3*ii];
        int p2 = fangle_idx[3*ii+1];
        int p3 = fangle_idx[3*ii+2];
        double k = fangle_params[2*ii];
        double theta = fangle_params[2*ii+1];
        // calc 2 vectors
        Vec3 d21,d23,d13;
        if (!ifPBC){
            d21 = ReferenceForce::getDeltaR(pos[p2], pos[p1]);
            d23 = ReferenceForce::getDeltaR(pos[p2], pos[p3]);
            d13 = ReferenceForce::getDeltaR(pos[p1], pos[p3]);
        } else {
            d21 = ReferenceForce::getDeltaRPeriodic(pos[p2], pos[p1], box);
            d23 = ReferenceForce::getDeltaRPeriodic(pos[p2], pos[p3], box);
            d13 = ReferenceForce::getDeltaRPeriodic(pos[p1], pos[p3], box);
        }
        double r21_2 = d21[0] * d21[0] + d21[1] * d21[1] + d21[2] * d21[2];
        double r23_2 = d23[0] * d23[0] + d23[1] * d23[1] + d23[2] * d23[2];
        double r13_2 = d13[0] * d13[0] + d13[1] * d13[1] + d13[2] * d13[2];
        double r21 = sqrt(r21_2);
        double r23 = sqrt(r23_2);
        double r13 = sqrt(r13_2);
        // calc angle
        double cost = (r23_2 + r21_2 - r13_2) / 2 / r21 / r23;
        double angle = acos(cost);
        // do something for dq
        double dq = k * (angle - theta);
        realcharges[p1] += dq;
        realcharges[p3] += dq;
        realcharges[p2] -= 2 * dq;

        // do something for dqdx
        // dp1/dp1: - \frac{k \left(\frac{cos(\theta) \left(- p_{1 x} + p_{2 x}\right)}{r_{21}^{2}} + \frac{- p_{2 x} + p_{3 x}}{r_{21} r_{23}}\right)}{\sqrt{1 - cos(\theta)^{2}}}
        // dp1/dp2: - dp1/dp1 - dp1/dp3
        // dp1/dp3: - \frac{k \left(\frac{cos(\theta) \left(p_{2 x} - p_{3 x}\right)}{r_{23}^{2}} + \frac{p_{1 x} - p_{2 x}}{r_{21} r_{23}}\right)}{\sqrt{1 - cos(\theta)^{2}}}
        // dp2/dp1: - dp1/dp1 - dp3/dp1
        // dp2/dp2: - dp2/dp1 - dp2/dp3
        // dp2/dp3: - dp1/dp3 - dp3/dp3
        // dp3/dp1: dp1/dp1
        // dp3/dp2: - dp3/dp1 - dp3/dp3
        // dp3/dp3: dp1/dp3

        // consts
        double one_r21r23 = 1.0 / r21 / r23;
        double one_const = 1 / sqrt(1 - cost * cost);

        double fin_const1 = k * one_r21r23 * one_const;
        double fin_const2_r21 = k * cost * one_const / r21_2;
        double fin_const2_r23 = k * cost * one_const / r23_2;

        // pair: 4 * numCFBonds + 9 * ii
        int pair1 = 4 * numCFBonds + 9 * ii;
        int pair2 = 4 * numCFBonds + 9 * ii + 1;
        int pair3 = 4 * numCFBonds + 9 * ii + 2;
        int pair4 = 4 * numCFBonds + 9 * ii + 3;
        int pair5 = 4 * numCFBonds + 9 * ii + 4;
        int pair6 = 4 * numCFBonds + 9 * ii + 5;
        int pair7 = 4 * numCFBonds + 9 * ii + 6;
        int pair8 = 4 * numCFBonds + 9 * ii + 7;
        int pair9 = 4 * numCFBonds + 9 * ii + 8;

        // val: 3*pair, 3*pair+1, 3*pair+2
        for(int jj=0;jj<3;jj++){
            double v1 = - fin_const1 * d23[jj] + fin_const2_r21 * d21[jj];
            double v3 = - fin_const1 * d21[jj] + fin_const2_r23 * d23[jj];
            double v2 = - v1 - v3;
            dqdx_val[3*pair1+jj] = v1;
            dqdx_val[3*pair2+jj] = v2;
            dqdx_val[3*pair3+jj] = v3;
            dqdx_val[3*pair4+jj] = - 2 * v1;
            dqdx_val[3*pair5+jj] = - 2 * v2;
            dqdx_val[3*pair6+jj] = - 2 * v3;
            dqdx_val[3*pair7+jj] = v1;
            dqdx_val[3*pair8+jj] = v2;
            dqdx_val[3*pair9+jj] = v3;
        }
    }
}

void ReferenceCalcCoulForceKernel::initialize(const System& system, const CoulForce& force) {
    int numParticles = system.getNumParticles();
    charges.resize(numParticles);
    for(int i=0;i<numParticles;i++){
        charges[i] = force.getParticleCharge(i);
    }
    realcharges.resize(numParticles);
    numCFBonds = force.getNumFluxBonds();
    fbond_idx.resize(numCFBonds*2);
    fbond_params.resize(numCFBonds*2);
    for(int ii=0;ii<numCFBonds;ii++){
        int p1, p2;
        double k, b;
        force.getFluxBondParameters(ii, p1, p2, k, b);
        fbond_idx[2*ii] = p1;
        fbond_idx[2*ii+1] = p2;
        fbond_params[2*ii] = k;
        fbond_params[2*ii+1] = b;
    }

    numCFAngles = force.getNumFluxAngles();
    fangle_idx.resize(numCFAngles*3);
    fangle_params.resize(numCFAngles*2);
    for(int ii=0;ii<numCFAngles;ii++){
        int p1, p2, p3;
        double k, theta;
        force.getFluxAngleParameters(ii, p1, p2, p3, k, theta);
        fangle_idx[3*ii] = p1;
        fangle_idx[3*ii+1] = p2;
        fangle_idx[3*ii+2] = p3;
        fangle_params[2*ii] = k;
        fangle_params[2*ii+1] = theta;
    }

    for(int ii=0;ii<numCFBonds;ii++){
        int p1, p2;
        double k, b;
        force.getFluxBondParameters(ii, p1, p2, k, b);
        // p1-p1
        dqdx_dqidx.push_back(p1);
        dqdx_dxidx.push_back(p1);

        // p1-p2
        dqdx_dqidx.push_back(p1);
        dqdx_dxidx.push_back(p2);

        // p2-p1
        dqdx_dqidx.push_back(p2);
        dqdx_dxidx.push_back(p1);

        // p2-p2
        dqdx_dqidx.push_back(p2);
        dqdx_dxidx.push_back(p2);

        for(int jj=0;jj<12;jj++){
            dqdx_val.push_back(0);
        }
    }

    for(int ii=0;ii<numCFAngles;ii++){
        int p1, p2, p3;
        double k, theta;
        force.getFluxAngleParameters(ii, p1, p2, p3, k, theta);
        // p1-p1
        dqdx_dqidx.push_back(p1);
        dqdx_dxidx.push_back(p1);
        // p1-p2
        dqdx_dqidx.push_back(p1);
        dqdx_dxidx.push_back(p2);
        // p1-p3
        dqdx_dqidx.push_back(p1);
        dqdx_dxidx.push_back(p3);
        // p2-p1
        dqdx_dqidx.push_back(p2);
        dqdx_dxidx.push_back(p1);
        // p2-p2
        dqdx_dqidx.push_back(p2);
        dqdx_dxidx.push_back(p2);
        // p2-p3
        dqdx_dqidx.push_back(p2);
        dqdx_dxidx.push_back(p3);
        // p3-p1
        dqdx_dqidx.push_back(p3);
        dqdx_dxidx.push_back(p1);
        // p3-p2
        dqdx_dqidx.push_back(p3);
        dqdx_dxidx.push_back(p2);
        // p3-p3
        dqdx_dqidx.push_back(p3);
        dqdx_dxidx.push_back(p3);

        for(int jj=0;jj<27;jj++){
            dqdx_val.push_back(0);
        }
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
    updateRealCharge(pos, box);
    double energy = 0.0;    
    double dEdR;
    vector<double> dedq;
    dedq.resize(numParticles);
    vector<double> deltaR;
    deltaR.resize(5);
    if (!ifPBC){
        // noPBC, calcall
        for(int ii=0;ii<numParticles;ii++){
            for(int jj=ii+1;jj<numParticles;jj++){
                ReferenceForce::getDeltaR(pos[ii], pos[jj], &deltaR[0]);
                double inverseR = 1.0 / deltaR[4];
                if (includeEnergy) {
                    energy += ONE_4PI_EPS0*realcharges[ii]*realcharges[jj]*inverseR;
                }
                if (includeForces) {
                    dEdR = ONE_4PI_EPS0*realcharges[ii]*realcharges[jj]*inverseR*inverseR*inverseR;
                    for(int dd=0;dd<3;dd++){
                        forces[ii][dd] -= dEdR*deltaR[dd];
                        forces[jj][dd] += dEdR*deltaR[dd];
                    }
                    dedq[ii] += ONE_4PI_EPS0*realcharges[jj]*inverseR;
                    dedq[jj] += ONE_4PI_EPS0*realcharges[ii]*inverseR;
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
                        energy -= ONE_4PI_EPS0*realcharges[p1]*realcharges[p2]*inverseR;
                    }
                    if (includeForces) {
                        dEdR = ONE_4PI_EPS0*realcharges[p1]*realcharges[p2]*inverseR*inverseR*inverseR;
                        for(int dd=0;dd<3;dd++){
                            forces[p1][dd] += dEdR*deltaR[dd];
                            forces[p2][dd] -= dEdR*deltaR[dd];
                        }
                    dedq[p1] -= ONE_4PI_EPS0*realcharges[p2]*inverseR;
                    dedq[p2] -= ONE_4PI_EPS0*realcharges[p1]*inverseR;
                    }
                }
            }
        }
        // calc dEdQ * dQdX
        for(int ii=0;ii<dqdx_dqidx.size();ii++){
            int p1 = dqdx_dqidx[ii];
            int p2 = dqdx_dxidx[ii];
            for(int jj=0;jj<3;jj++){
                forces[p2][jj] -= dedq[p1] * dqdx_val[3*ii+jj];
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
            selfEwaldEnergy -= ONE_4PI_EPS0 * realcharges[ii] * realcharges[ii] * alpha / sqrt(M_PI);
            dedq[ii] += - 2 * ONE_4PI_EPS0 * alpha / sqrt(M_PI) * realcharges[ii];
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
                            cs += realcharges[ii] * cos(gr);
                            ss += realcharges[ii] * sin(gr);
                        }
                    }
                    if(includeForces){
                        for(int ii=0;ii<numParticles;ii++){
                            double gr = kx * pos[ii][0] + ky * pos[ii][1] + kz * pos[ii][2];
                            double gradr = 2.0 * constant * eak * (ss * realcharges[ii] * cos(gr) - cs * realcharges[ii] * sin(gr));
                            forces[ii][0] -= gradr * kx;
                            forces[ii][1] -= gradr * ky;
                            forces[ii][2] -= gradr * kz;

                            dedq[ii] += 2 * constant * eak * (cs * cos(gr) + ss * sin(gr));
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
                double dEdR = ONE_4PI_EPS0 * realcharges[ii] * realcharges[jj] * inverseR * inverseR * inverseR;
                dEdR = dEdR * (erfc(alphaR) + alphaR * exp (- alphaR * alphaR) * 2.0 / sqrt(M_PI));
                for(int kk=0;kk<3;kk++){
                    double fconst = dEdR*deltaR[0][kk];
                    forces[ii][kk] += fconst;
                    forces[jj][kk] -= fconst;
                }
                dedq[ii] += ONE_4PI_EPS0*realcharges[jj]*inverseR*erfc(alphaR);
                dedq[jj] += ONE_4PI_EPS0*realcharges[ii]*inverseR*erfc(alphaR);
            }

            realSpaceEwaldEnergy += ONE_4PI_EPS0*realcharges[ii]*realcharges[jj]*inverseR*erfc(alphaR);
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
                        double dEdR = ONE_4PI_EPS0 * realcharges[p1] * realcharges[p2] * inverseR * inverseR * inverseR;
                        dEdR = dEdR * (erf(alphaR) - alphaR * exp (- alphaR * alphaR) * 2.0 / sqrt(M_PI));
                        for(int kk=0;kk<3;kk++){
                            double fconst = dEdR*deltaR[0][kk];
                            forces[p1][kk] -= fconst;
                            forces[p2][kk] += fconst;
                        }
                    dedq[p1] -= ONE_4PI_EPS0*realcharges[p2]*inverseR*erf(alphaR);
                    dedq[p2] -= ONE_4PI_EPS0*realcharges[p1]*inverseR*erf(alphaR);
                    }

                    realSpaceException -= ONE_4PI_EPS0*realcharges[p1]*realcharges[p2]*inverseR*erf(alphaR);
                }
            }
        }
        

        // calc dEdQ * dQdX
        for(int ii=0;ii<dqdx_dqidx.size();ii++){
            int p1 = dqdx_dqidx[ii];
            int p2 = dqdx_dxidx[ii];
            for(int jj=0;jj<3;jj++){
                forces[p2][jj] -= dedq[p1] * dqdx_val[3*ii+jj];
            }
        }
        energy = selfEwaldEnergy + recipEnergy + realSpaceEwaldEnergy + realSpaceException;
    } 
    return energy;
}