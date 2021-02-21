#ifndef OPENMM_COULFORCE_H_
#define OPENMM_COULFORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <string>
#include <utility>
#include <vector>

namespace CoulPlugin {

/**
 * This class implements Coul-kit force field. 
 */

class CoulForce : public OpenMM::Force {
public:
    /**
     * Create a CoulForce.  The network is defined by a TensorFlow graph saved
     * to a binary protocol buffer file.
     */
    CoulForce();
    /**
     * Get the pre-factor for cos accelerate force.
     * @param charge    charge of a particle
     */
    void addParticle(double charge, double sigma, double epsilon);
    /**
     * Get the number of particles in the system
     * @return          the number of particle
     */
    int getNumParticles() const;
    /**
     * Get the charge of specific particle
     * @return          the charge of particle
     */
    void getParticleParameters(int index, double& charge, double& sigma, double& epsilon) const;
    /**
     * Set charge of a specific particle
     * @param index     the index of particle
     * @param charge    the charge of particle
     */
    void setParticleParameters(int index, double charge, double sigma, double epsilon);
    /**
     * Get the cutoff of this system. Will not be used in noPBC system.
     * @return           the cutoff value
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff of this system. Will not be used in noPBC system.
     * @param cutoff      the cutoff value
     */
    void setCutoffDistance(double cutoff);
    /**
     * Return if PBC is used in this force. Default is no.
     * @return             whether PBC system
     */
    bool usesPeriodicBoundaryConditions() const;
    /**
     * Set if using PBC in the system.
     * @param ifPeriod     if use PBC
     */
    void setUsesPeriodicBoundaryConditions(bool ifPeriod);
    /**
     * Add exception pair. The short range interaction between this pair
     * would not be calculated.
     * @param p1            the index of particle1
     * @param p2            the index of particle2
     */
    void addException(int p1, int p2);
    /**
     * Get the total number of exceptions.
     * @return              the number of exceptions
     */
    int getNumExceptions() const;
    /**
     * Get the indexes of specific exception
     * @param index          the index of exception we need
     * @param p1             output index 1
     * @param p2             output index 2
     */
    void getExceptionParameters(const int index, int& p1, int& p2) const;
    /**
     * Set the tolerance of Ewald sum. It will be used to generate k-vectors
     * for reciprocal space.
     * @param tol             Ewald tollerance
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Get the tolerance of Ewald Summation
     * @return                Ewald tolerance
     */
    double getEwaldErrorTolerance() const;
    /**
     * Add a charge-flux bond
     * @param p1
     * @param p2
     * @param k
     * @param b
     */
    void addFluxBond(int p1, int p2, double k, double b);
    /**
     * Get parameter of charge-flux bond
     */
    void getFluxBondParameters(int index, int& p1, int& p2, double& k, double& b) const;
    /**
     * Get number of charge-flux bond
     */
    int getNumFluxBonds() const;
    /**
     * Add charge-flux angle
     */
    void addFluxAngle(int p1, int p2, int p3, double k, double theta);
    /**
     * Get parameters of charge-flux angle
     */
    void getFluxAngleParameters(int index, int& p1, int& p2, int& p3, double& k, double& theta) const;
    /**
     * Get number of charge-flux angle
     */
    int getNumFluxAngles() const;
    /**
     * Add flux water. 
     */
    void addFluxWater(int po, int ph1, int ph2, double k1, double k2, double kub, double b0, double ub0);
    /**
     * Get parameter of flux water.
     */
    void getFluxWaterParameters(int index, int& po, int& ph1, int& ph2, double& k1, double& k2, double& kub, double& b0, double& ub0) const;
    /**
     * Get number of flux water. 
     */
    int getNumFluxWater() const;

protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    std::vector<std::pair<int,int>> exclusions;
    std::vector<double> charges;
    std::vector<int> fbond_idx;
    std::vector<double> fbond_params;
    std::vector<int> fangle_idx;
    std::vector<double> fangle_params;
    std::vector<double> ljparams;
    std::vector<int> fwater_idx;
    std::vector<double> fwater_params;
    double cutoffDistance;
    double ewaldTol;
    bool ifPBC;
};

} // namespace CoulPlugin

#endif /*OPENMM_COULFORCE_H_*/