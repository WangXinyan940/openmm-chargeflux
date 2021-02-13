#ifndef OPENMM_CUDAKERNELSOURCES_H_
#define OPENMM_CUDAKERNELSOURCES_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2012 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include <string>

namespace OpenMM {

/**
 * This class is a central holding place for the source code of CUDA kernels.
 * The CMake build script inserts declarations into it based on the .cu files in the
 * kernels subfolder.
 */

class CudaKernelSources {
public:
static const std::string gbsaObc1;
static const std::string findInteractingBlocks;
static const std::string nonbonded;
static const std::string customNonbondedGroups;
static const std::string coulombLennardJones;
static const std::string nonbondedExceptions;
static const std::string monteCarloBarostat;
static const std::string ewald;
static const std::string removeCM;
static const std::string customGBEnergyN2;
static const std::string vectorOps;
static const std::string langevin;
static const std::string brownian;
static const std::string bondForce;
static const std::string rbTorsionForce;
static const std::string customIntegrator;
static const std::string customGBValueN2;
static const std::string harmonicBondForce;
static const std::string customIntegratorGlobal;
static const std::string torsionForce;
static const std::string periodicTorsionForce;
static const std::string customNonbonded;
static const std::string utilities;
static const std::string customGBValuePerParticle;
static const std::string parallel;
static const std::string customIntegratorPerDof;
static const std::string customGBEnergyPerParticle;
static const std::string pme;
static const std::string customHbondForce;
static const std::string verlet;
static const std::string sort;
static const std::string customGBChainRule;
static const std::string angleForce;
static const std::string integrationUtilities;
static const std::string customCompoundBond;
static const std::string customManyParticle;
static const std::string gbsaObc2;
static const std::string customExternalForce;
static const std::string constraints;
static const std::string customGBGradientChainRule;
static const std::string cmapTorsionForce;
static const std::string harmonicAngleForce;
static const std::string andersenThermostat;

};

} // namespace OpenMM

#endif /*OPENMM_CUDAKERNELSOURCES_H_*/