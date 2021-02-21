#include "CoulForce.h"
#include "internal/CoulForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>
#include <cmath>

using namespace CoulPlugin;
using namespace OpenMM;
using namespace std;

CoulForce::CoulForce() {
    cutoffDistance = 1.0;
    ewaldTol = 0.0001;
    ifPBC = false;
}

void CoulForce::addParticle(double charge, double sigma, double epsilon){
    charges.push_back(charge);
    ljparams.push_back(sigma);
    ljparams.push_back(epsilon);
}

int CoulForce::getNumParticles() const {
    return charges.size();
}

void CoulForce::getParticleParameters(int index, double& charge, double& sigma, double& epsilon) const {
    charge = charges[index];
    sigma = ljparams[2*index];
    epsilon = ljparams[2*index+1];
}

void CoulForce::setParticleParameters(int index, double charge, double sigma, double epsilon) {
    charges[index] = charge;
    ljparams[2*index] = sigma;
    ljparams[2*index+1] = epsilon;
}

double CoulForce::getCutoffDistance() const {
    return cutoffDistance;
}

void CoulForce::setCutoffDistance(double cutoff){
    cutoffDistance = cutoff;
}

bool CoulForce::usesPeriodicBoundaryConditions() const {
    return ifPBC;
}

void CoulForce::setUsesPeriodicBoundaryConditions(bool ifPeriod){
    ifPBC = ifPeriod;
}

void CoulForce::addException(int p1, int p2){
    pair<int,int> expair(p1, p2);
    exclusions.push_back(expair);
}

int CoulForce::getNumExceptions() const {
    return exclusions.size();
}

void CoulForce::getExceptionParameters(const int index, int& p1, int& p2) const {
    p1 = exclusions[index].first;
    p2 = exclusions[index].second;
}

void CoulForce::setEwaldErrorTolerance(double tol){
    ewaldTol = tol;
}

double CoulForce::getEwaldErrorTolerance() const{
    return ewaldTol;
}

void CoulForce::addFluxBond(int p1, int p2, double k, double b){
    fbond_idx.push_back(p1);
    fbond_idx.push_back(p2);
    fbond_params.push_back(k);
    fbond_params.push_back(b);
}

void CoulForce::getFluxBondParameters(int index, int& p1, int& p2, double& k, double& b) const{
    p1 = fbond_idx[2*index];
    p2 = fbond_idx[2*index+1];
    k = fbond_params[2*index];
    b = fbond_params[2*index+1];
}

int CoulForce::getNumFluxBonds() const{
    return fbond_idx.size()/2;
}

void CoulForce::addFluxAngle(int p1, int p2, int p3, double k, double theta){
    fangle_idx.push_back(p1);
    fangle_idx.push_back(p2);
    fangle_idx.push_back(p3);
    fangle_params.push_back(k);
    fangle_params.push_back(theta);
}

void CoulForce::getFluxAngleParameters(int index, int& p1, int& p2, int& p3, double& k, double& theta) const{
    p1 = fangle_idx[3*index];
    p2 = fangle_idx[3*index+1];
    p3 = fangle_idx[3*index+2];
    k = fangle_params[2*index];
    theta = fangle_params[2*index+1];
}

int CoulForce::getNumFluxAngles() const{
    return fangle_idx.size() / 3;
}

void CoulForce::addFluxWater(int po, int ph1, int ph2, double k1, double k2, double kub, double b0, double ub0){
    fwater_idx.push_back(po);
    fwater_idx.push_back(ph1);
    fwater_idx.push_back(ph2);
    fwater_params.push_back(k1);
    fwater_params.push_back(k2);
    fwater_params.push_back(kub);
    fwater_params.push_back(b0);
    fwater_params.push_back(ub0);
}

void CoulForce::getFluxWaterParameters(int index, int& po, int& ph1, int& ph2, double& k1, double& k2, double& kub, double& b0, double& ub0) const {
    po = fwater_idx[3*index];
    ph1 = fwater_idx[3*index+1];
    ph2 = fwater_idx[3*index+2];
    k1 = fwater_params[5*index];
    k2 = fwater_params[5*index+1];
    kub = fwater_params[5*index+2];
    b0 = fwater_params[5*index+3];
    ub0 = fwater_params[5*index+4];
}

int CoulForce::getNumFluxWaters() const {
    return fwater_idx.size() / 3;
}

ForceImpl* CoulForce::createImpl() const {
    return new CoulForceImpl(*this);
}

