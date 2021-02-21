  
%module openmmcoul

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_vector.i>

%{
#include "CoulForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

%feature("shadow") CoulPlugin::CoulForce::CoulForce %{
    def __init__(self, *args):
        this = _openmmcoul.new_CoulForce()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
%}

namespace std {
  %template(IntVector) vector<int>;
}

namespace CoulPlugin {

class CoulForce : public OpenMM::Force {
public:
    CoulForce();
    void addParticle(double charge, double sigma, double epsilon);
    int getNumParticles() const;
    void setParticleParameters(int index, double charge, double sigma, double epsilon);
    double getCutoffDistance() const;
    void setCutoffDistance(double cutoff);
    void setUsesPeriodicBoundaryConditions(bool ifPeriod);
    void addException(int p1, int p2);
    int getNumExceptions() const;
    void setEwaldErrorTolerance(double tol);
    double getEwaldErrorTolerance() const;
    void addFluxBond(int p1, int p2, double k, double b);
    void addFluxAngle(int p1, int p2, int p3, double k, double theta);
    void addFluxWater(int po, int ph1, int ph2, double k1, double k2, double kub, double b0, double ub0);
    
    %extend {
        static CoulPlugin::CoulForce& cast(OpenMM::Force& force) {
            return dynamic_cast<CoulPlugin::CoulForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<CoulPlugin::CoulForce*>(&force) != NULL);
        }
    }
};

}