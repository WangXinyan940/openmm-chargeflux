#include "CoulForce.h"
#include "internal/CoulForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>

using namespace CoulPlugin;
using namespace OpenMM;
using namespace std;

CoulForce::CoulForce() {
    cutoffDistance = 1.0;
    ewaldTol = 0.0001;
    ifPBC = false;
}

void CoulForce::addParticle(double charge){
    charges.push_back(charge);
}

int CoulForce::getNumParticles() const {
    return charges.size();
}

double CoulForce::getParticleCharge(int index) const {
    return charges[index];
}

void CoulForce::setParticleCharge(int index, double charge){
    charges[index] = charge;
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

ForceImpl* CoulForce::createImpl() const {
    return new CoulForceImpl(*this);
}

