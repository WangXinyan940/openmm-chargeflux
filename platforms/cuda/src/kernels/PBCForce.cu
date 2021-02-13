#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

#define COMPUTE_INTERACTION \
tempEnergy += ONE_4PI_EPS0 * atomData1.chrg * atomData2.chrg * invR * erfcAlphaR;\
dEdR += ONE_4PI_EPS0 * atomData1.chrg * atomData2.chrg * invR * invR * invR;\
dEdR = dEdR * (erfcAlphaR  + alphaR * EXP(- alphaR * alphaR) * TWO_OVER_SQRT_PI)

typedef struct {
    real x, y, z, q;
    real chrg;
    real fx, fy, fz;
} AtomData;


/**
 * Compute nonbonded interactions. The kernel is separated into two parts,
 * tiles with exclusions and tiles without exclusions. It relies heavily on 
 * implicit warp-level synchronization. A tile is defined by two atom blocks 
 * each of warpsize. Each warp computes a range of tiles.
 * 
 * Tiles with exclusions compute the entire set of interactions across
 * atom blocks, equal to warpsize*warpsize. In order to avoid access conflicts 
 * the forces are computed and accumulated diagonally in the manner shown below
 * where, suppose
 *
 * [a-h] comprise atom block 1, [i-p] comprise atom block 2
 *
 * 1 denotes the first set of calculations within the warp
 * 2 denotes the second set of calculations within the warp
 * ... etc.
 * 
 *        threads
 *     0 1 2 3 4 5 6 7
 *         atom1 
 * L    a b c d e f g h 
 * o  i 1 2 3 4 5 6 7 8
 * c  j 8 1 2 3 4 5 6 7
 * a  k 7 8 1 2 3 4 5 6
 * l  l 6 7 8 1 2 3 4 5
 * D  m 5 6 7 8 1 2 3 4 
 * a  n 4 5 6 7 8 1 2 3
 * t  o 3 4 5 6 7 8 1 2
 * a  p 2 3 4 5 6 7 8 1
 *
 * Tiles without exclusions read off directly from the neighbourlist interactingAtoms
 * and follows the same force accumulation method. If more there are more interactingTiles
 * than the size of the neighbourlist initially allocated, the neighbourlist is rebuilt
 * and the full tileset is computed. This should happen on the first step, and very rarely 
 * afterwards.
 *
 * On CUDA devices that support the shuffle intrinsic, on diagonal exclusion tiles use
 * __shfl to broadcast. For all other types of tiles __shfl is used to pass around the 
 * forces, positions, and parameters when computing the forces. 
 *
 * [out]forceBuffers    - forces on each atom to eventually be accumulated
 * [out]energyBuffer    - energyBuffer to eventually be accumulated
 * [in]posq             - x,y,z,charge 
 * [in]exclusions       - 1024-bit flags denoting atom-atom exclusions for each tile
 * [in]exclusionTiles   - x,y denotes the indices of tiles that have an exclusion
 * [in]startTileIndex   - index into first tile to be processed
 * [in]numTileIndices   - number of tiles this context is responsible for processing
 * [in]int tiles        - the atom block for each tile
 * [in]interactionCount - total number of tiles that have an interaction
 * [in]maxTiles         - stores the size of the neighbourlist in case it needs 
 *                      - to be expanded
 * [in]periodicBoxSize  - size of the Periodic Box, last dimension (w) not used
 * [in]invPeriodicBox   - inverse of the periodicBoxSize, pre-computed for speed
 * [in]blockCenter      - the center of each block in euclidean coordinates
 * [in]blockSize        - size of the each block, radiating from the center
 *                      - x is half the distance of total length
 *                      - y is half the distance of total width
 *                      - z is half the distance of total height
 *                      - w is not used
 * [in]interactingAtoms - a list of interactions within a given tile     
 *
 */
extern "C" __global__ void computeNonbonded(
        unsigned long long*       __restrict__     forceBuffers, 
        mixed*                    __restrict__     energyBuffer, 
        const real4*              __restrict__     posq, 
        const real*               __restrict__     charges,
        const int*                __restrict__     atomIndex,
        const tileflags*          __restrict__     exclusions,
        const int2*               __restrict__     exclusionTiles, 
        unsigned int                               startTileIndex, 
        unsigned long long                         numTileIndices,
        const int*                __restrict__     tiles, 
        const unsigned int*       __restrict__     interactionCount, 
        real4                                      periodicBoxSize, 
        real4                                      invPeriodicBoxSize, 
        real4                                      periodicBoxVecX, 
        real4                                      periodicBoxVecY, 
        real4                                      periodicBoxVecZ, 
        unsigned int                               maxTiles, 
        const real4*              __restrict__     blockCenter,
        const real4*              __restrict__     blockSize, 
        const unsigned int*       __restrict__     interactingAtoms, 
        unsigned int                               maxSinglePairs,
        const int2*               __restrict__     singlePairs
        ) {
    const unsigned int totalWarps = (blockDim.x*gridDim.x)/TILE_SIZE;
    const unsigned int warp = (blockIdx.x*blockDim.x+threadIdx.x)/TILE_SIZE; // global warpIndex
    const unsigned int tgx = threadIdx.x & (TILE_SIZE-1); // index within the warp
    const unsigned int tbx = threadIdx.x - tgx;           // block warpIndex
    mixed energy = 0;
    // used shared memory if the device cannot shuffle
    __shared__ AtomData localData[THREAD_BLOCK_SIZE];

    real cutoff2 = CUTOFF * CUTOFF;

    // First loop: process tiles that contain exclusions.

    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const int2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        real3 force = make_real3(0);
        unsigned int atom1 = x*TILE_SIZE + tgx;
        real4 posq1 = posq[atom1];
        
        // LOAD_ATOM1_PARAMETERS
        AtomData atomData1;
        atomData1.x = posq1.x;
        atomData1.y = posq1.y;
        atomData1.z = posq1.z;
        atomData1.chrg = charges[atomIndex[atom1]];

#ifdef USE_EXCLUSIONS
        tileflags excl = exclusions[pos*TILE_SIZE+tgx];
#endif
        const bool hasExclusions = true;
        if (x == y) {
            // This tile is on the diagonal.
            localData[threadIdx.x].x = posq1.x;
            localData[threadIdx.x].y = posq1.y;
            localData[threadIdx.x].z = posq1.z;
            localData[threadIdx.x].chrg = charges[atomIndex[atom1]];

            // we do not need to fetch parameters from global since this is a symmetric tile
            // instead we can broadcast the values using shuffle
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+j;
                real3 posq2;

                posq2 = make_real3(localData[atom2].x, localData[atom2].y, localData[atom2].z);
                real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                real invR = RSQRT(r2);
                real r = r2*invR;
                
                // LOAD_ATOM2_PARAMETERS
                AtomData atomData2;
                atomData2.x = posq2.x;
                atomData2.y = posq2.y;
                atomData2.z = posq2.z;
                atomData2.chrg = localData[atom2].chrg;

                atom2 = y*TILE_SIZE+j;
#ifdef USE_SYMMETRIC
                real dEdR = 0.0f;
#else
                real3 dEdR1 = make_real3(0);
                real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & 0x1));
#endif
                real tempEnergy = 0.0f;
                const real interactionScale = 0.5f;
                if (!isExcluded && r2 < cutoff2){
                    real alphaR = EWALD_ALPHA * r;
#ifdef USE_DOUBLE_PRECISION
                    const real erfcAlphaR = erfc(alphaR);
#else
                    const real t = RECIP(1.0f+0.3275911f*alphaR);
                    const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
                    COMPUTE_INTERACTION;
                }
                energy += 0.5f*tempEnergy;
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
                force.x -= delta.x*dEdR;
                force.y -= delta.y*dEdR;
                force.z -= delta.z*dEdR;
#else
                force.x -= dEdR1.x;
                force.y -= dEdR1.y;
                force.z -= dEdR1.z;
#endif
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif
            }
        }
        else {
            // This is an off-diagonal tile.
            unsigned int j = y*TILE_SIZE + tgx;
            real4 shflPosq = posq[j];

            localData[threadIdx.x].x = shflPosq.x;
            localData[threadIdx.x].y = shflPosq.y;
            localData[threadIdx.x].z = shflPosq.z;
            localData[threadIdx.x].fx = 0.0f;
            localData[threadIdx.x].fy = 0.0f;
            localData[threadIdx.x].fz = 0.0f;
            localData[threadIdx.x].chrg = charges[atomIndex[j]];

            
            
#ifdef USE_EXCLUSIONS
            excl = (excl >> tgx) | (excl << (TILE_SIZE - tgx));
#endif
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+tj;

                real3 posq2 = make_real3(localData[atom2].x, localData[atom2].y, localData[atom2].z);

                real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                real invR = RSQRT(r2);
                real r = r2*invR;
                
                // LOAD_ATOM2_PARAMETERS
                AtomData atomData2;
                atomData2.x = posq2.x;
                atomData2.y = posq2.y;
                atomData2.z = posq2.z;
                atomData2.chrg = localData[atom2].chrg;

                atom2 = y*TILE_SIZE+tj;
#ifdef USE_SYMMETRIC
                real dEdR = 0.0f;
#else
                real3 dEdR1 = make_real3(0);
                real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & 0x1));
#endif
                real tempEnergy = 0.0f;
                const real interactionScale = 1.0f;
                if (!isExcluded && r2 < cutoff2){
                    real alphaR = EWALD_ALPHA * r;
#ifdef USE_DOUBLE_PRECISION
                    const real erfcAlphaR = erfc(alphaR);
#else
                    const real t = RECIP(1.0f+0.3275911f*alphaR);
                    const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
                    COMPUTE_INTERACTION;
                }
                energy += tempEnergy;
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
                delta *= dEdR;
                force.x -= delta.x;
                force.y -= delta.y;
                force.z -= delta.z;

                localData[tbx+tj].fx += delta.x;
                localData[tbx+tj].fy += delta.y;
                localData[tbx+tj].fz += delta.z;

#else // !USE_SYMMETRIC
                force.x -= dEdR1.x;
                force.y -= dEdR1.y;
                force.z -= dEdR1.z;

                localData[tbx+tj].fx += dEdR2.x;
                localData[tbx+tj].fy += dEdR2.y;
                localData[tbx+tj].fz += dEdR2.z;

#endif // end USE_SYMMETRIC
#endif

#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif
                // cycles the indices
                // 0 1 2 3 4 5 6 7 -> 1 2 3 4 5 6 7 0
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            const unsigned int offset = y*TILE_SIZE + tgx;
            // write results for off diagonal tiles
#ifdef INCLUDE_FORCES

            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fx*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fy*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fz*0x100000000)));

#endif
        }
        // Write results for on and off diagonal tiles
#ifdef INCLUDE_FORCES
        const unsigned int offset = x*TILE_SIZE + tgx;
        atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
#endif
    }

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

#ifdef USE_CUTOFF
    const unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return; // There wasn't enough memory for the neighbor list.
    int pos = (int) (warp*(long long)numTiles/totalWarps);
    int end = (int) ((warp+1)*(long long)numTiles/totalWarps);
#else
    int pos = (int) (startTileIndex+warp*numTileIndices/totalWarps);
    int end = (int) (startTileIndex+(warp+1)*numTileIndices/totalWarps);
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
    // atomIndices can probably be shuffled as well
    // but it probably wouldn't make things any faster
    __shared__ int atomIndices[THREAD_BLOCK_SIZE];
    __shared__ volatile int skipTiles[THREAD_BLOCK_SIZE];
    skipTiles[threadIdx.x] = -1;
    
    while (pos < end) {
        const bool hasExclusions = false;
        real3 force = make_real3(0);
        bool includeTile = true;

        // Extract the coordinates of this tile.
        int x, y;
        bool singlePeriodicCopy = false;
#ifdef USE_CUTOFF
        x = tiles[pos];
        real4 blockSizeX = blockSize[x];
        singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= CUTOFF &&
                              0.5f*periodicBoxSize.y-blockSizeX.y >= CUTOFF &&
                              0.5f*periodicBoxSize.z-blockSizeX.z >= CUTOFF);
#else
        y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
        x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
            y += (x < y ? -1 : 1);
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        }

        // Skip over tiles that have exclusions, since they were already processed.

        while (skipTiles[tbx+TILE_SIZE-1] < pos) {
            if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
                int2 tile = exclusionTiles[skipBase+tgx];
                skipTiles[threadIdx.x] = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
            }
            else
                skipTiles[threadIdx.x] = end;
            skipBase += TILE_SIZE;            
            currentSkipIndex = tbx;
        }
        while (skipTiles[currentSkipIndex] < pos)
            currentSkipIndex++;
        includeTile = (skipTiles[currentSkipIndex] != pos);
#endif
        if (includeTile) {
            unsigned int atom1 = x*TILE_SIZE + tgx;
            // Load atom data for this tile.
            real4 posq1 = posq[atom1];

            // LOAD_ATOM1_PARAMETERS
            AtomData atomData1;
            atomData1.x = posq1.x;
            atomData1.y = posq1.y;
            atomData1.z = posq1.z;
            atomData1.chrg = charges[atomIndex[atom1]];

            //const unsigned int localAtomIndex = threadIdx.x;
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[threadIdx.x] = j;

            if (j < PADDED_NUM_ATOMS) {
                // Load position of atom j from from global memory

                localData[threadIdx.x].x = posq[j].x;
                localData[threadIdx.x].y = posq[j].y;
                localData[threadIdx.x].z = posq[j].z;
                localData[threadIdx.x].fx = 0.0f;
                localData[threadIdx.x].fy = 0.0f;
                localData[threadIdx.x].fz = 0.0f;
                localData[threadIdx.x].chrg = charges[atomIndex[j]];
                
            }
            else {

                localData[threadIdx.x].x = 0;
                localData[threadIdx.x].y = 0;
                localData[threadIdx.x].z = 0;

                localData[threadIdx.x].chrg = 0.0;
            }
#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                // The box is small enough that we can just translate all the atoms into a single periodic
                // box, then skip having to apply periodic boundary conditions later.
                real4 blockCenterX = blockCenter[x];
                APPLY_PERIODIC_TO_POS_WITH_CENTER(posq1, blockCenterX)

                APPLY_PERIODIC_TO_POS_WITH_CENTER(localData[threadIdx.x], blockCenterX)

                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;

                    real3 posq2 = make_real3(localData[atom2].x, localData[atom2].y, localData[atom2].z);

                    real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    
                    // LOAD_ATOM2_PARAMETERS
                    AtomData atomData2;
                    atomData2.x = posq2.x;
                    atomData2.y = posq2.y;
                    atomData2.z = posq2.z;
                    atomData2.chrg = localData[atom2].chrg;

                    atom2 = atomIndices[tbx+tj];
#ifdef USE_SYMMETRIC
                    real dEdR = 0.0f;
#else
                    real3 dEdR1 = make_real3(0);
                    real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                    bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS);
#endif
                    real tempEnergy = 0.0f;
                    const real interactionScale = 1.0f;

                    if (!isExcluded && r2 < cutoff2){
                        real alphaR = EWALD_ALPHA * r;
#ifdef USE_DOUBLE_PRECISION
                        const real erfcAlphaR = erfc(alphaR);
#else
                        const real t = RECIP(1.0f+0.3275911f*alphaR);
                        const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
                        COMPUTE_INTERACTION;
                    }
                    
                    energy += tempEnergy;
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
                    delta *= dEdR;
                    force.x -= delta.x;
                    force.y -= delta.y;
                    force.z -= delta.z;

                    localData[tbx+tj].fx += delta.x;
                    localData[tbx+tj].fy += delta.y;
                    localData[tbx+tj].fz += delta.z;

#else // !USE_SYMMETRIC
                    force.x -= dEdR1.x;
                    force.y -= dEdR1.y;
                    force.z -= dEdR1.z;

                    localData[tbx+tj].fx += dEdR2.x;
                    localData[tbx+tj].fy += dEdR2.y;
                    localData[tbx+tj].fz += dEdR2.z;

#endif // end USE_SYMMETRIC
#endif

                    tj = (tj + 1) & (TILE_SIZE - 1);
                }
            }
            else
#endif
            {
                // We need to apply periodic boundary conditions separately for each interaction.
                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;

                    real3 posq2 = make_real3(localData[atom2].x, localData[atom2].y, localData[atom2].z);
                    real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    
                    // LOAD_ATOM2_PARAMETERS
                    AtomData atomData2;
                    atomData2.x = posq2.x;
                    atomData2.y = posq2.y;
                    atomData2.z = posq2.z;
                    atomData2.chrg = localData[atom2].chrg;

                    atom2 = atomIndices[tbx+tj];
#ifdef USE_SYMMETRIC
                    real dEdR = 0.0f;
#else
                    real3 dEdR1 = make_real3(0);
                    real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                    bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS);
#endif
                    real tempEnergy = 0.0f;
                    const real interactionScale = 1.0f;

                    if (!isExcluded && r2 < cutoff2){
                        real alphaR = EWALD_ALPHA * r;
#ifdef USE_DOUBLE_PRECISION
                        const real erfcAlphaR = erfc(alphaR);
#else
                        const real t = RECIP(1.0f+0.3275911f*alphaR);
                        const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
                        COMPUTE_INTERACTION;
                    }
                    
                    energy += tempEnergy;
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
                    delta *= dEdR;
                    force.x -= delta.x;
                    force.y -= delta.y;
                    force.z -= delta.z;

                    localData[tbx+tj].fx += delta.x;
                    localData[tbx+tj].fy += delta.y;
                    localData[tbx+tj].fz += delta.z;

#else // !USE_SYMMETRIC
                    force.x -= dEdR1.x;
                    force.y -= dEdR1.y;
                    force.z -= dEdR1.z;

                    localData[tbx+tj].fx += dEdR2.x;
                    localData[tbx+tj].fy += dEdR2.y;
                    localData[tbx+tj].fz += dEdR2.z;

#endif // end USE_SYMMETRIC
#endif

                    tj = (tj + 1) & (TILE_SIZE - 1);
                }
            }

            // Write results.
#ifdef INCLUDE_FORCES
            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
            atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
            atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
#ifdef USE_CUTOFF
            unsigned int atom2 = atomIndices[threadIdx.x];
#else
            unsigned int atom2 = y*TILE_SIZE + tgx;
#endif
            if (atom2 < PADDED_NUM_ATOMS) {

                atomicAdd(&forceBuffers[atom2], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fx*0x100000000)));
                atomicAdd(&forceBuffers[atom2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fy*0x100000000)));
                atomicAdd(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fz*0x100000000)));

            }
#endif
        }
        pos++;
    }
    
    // Third loop: single pairs that aren't part of a tile.
    
#if USE_CUTOFF
    const unsigned int numPairs = interactionCount[1];
    if (numPairs > maxSinglePairs)
        return; // There wasn't enough memory for the neighbor list.
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numPairs; i += blockDim.x*gridDim.x) {
        int2 pair = singlePairs[i];
        int atom1 = pair.x;
        int atom2 = pair.y;
        real4 posq1 = posq[atom1];
        real4 posq2 = posq[atom2];

        // LOAD_ATOM1_PARAMETERS
        AtomData atomData1;
        atomData1.x = posq1.x;
        atomData1.y = posq1.y;
        atomData1.z = posq1.z;
        atomData1.chrg = charges[atomIndex[atom1]];
        
        int j = atom2;
        // atom2 = threadIdx.x;
        
        // LOAD_ATOM2_PARAMETERS
        AtomData atomData2;
        atomData2.x = posq2.x;
        atomData2.y = posq2.y;
        atomData2.z = posq2.z;
        atomData2.chrg = charges[atomIndex[atom2]];
        
        // atom2 = pair.y;
        real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_DELTA(delta)
#endif
        real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        real invR = RSQRT(r2);
        real r = r2*invR;
#ifdef USE_SYMMETRIC
        real dEdR = 0.0f;
#else
        real3 dEdR1 = make_real3(0);
        real3 dEdR2 = make_real3(0);
#endif
        bool hasExclusions = false;
        bool isExcluded = false;
        real tempEnergy = 0.0f;
        const real interactionScale = 1.0f;

        if (!isExcluded && r2 < cutoff2){
            real alphaR = EWALD_ALPHA * r;
#ifdef USE_DOUBLE_PRECISION
            const real erfcAlphaR = erfc(alphaR);
#else
            const real t = RECIP(1.0f+0.3275911f*alphaR);
            const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
            COMPUTE_INTERACTION;
        }
        
        energy += tempEnergy;
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
        real3 dEdR1 = delta*dEdR;
        real3 dEdR2 = -dEdR1;
#endif
        atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long) (-dEdR1.x*0x100000000)));
        atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-dEdR1.y*0x100000000)));
        atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-dEdR1.z*0x100000000)));
        atomicAdd(&forceBuffers[atom2], static_cast<unsigned long long>((long long) (-dEdR2.x*0x100000000)));
        atomicAdd(&forceBuffers[atom2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-dEdR2.y*0x100000000)));
        atomicAdd(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-dEdR2.z*0x100000000)));
#endif
    }
#endif
#ifdef INCLUDE_ENERGY
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
#endif
}

extern "C" __global__ void computeExclusion(
    unsigned long long*       __restrict__     forceBuffers, 
    mixed*                    __restrict__     energyBuffer, 
    const real4*              __restrict__     posq, 
    const real*               __restrict__     charges,
    const int*                __restrict__     atomIndex,
    const int*                __restrict__     indexAtom,
    const int*                __restrict__     exclusionidx1,
    const int*                __restrict__     exclusionidx2,
    const int                                  numExclusions,
    real4                                      periodicBoxSize, 
    real4                                      invPeriodicBoxSize, 
    real4                                      periodicBoxVecX, 
    real4                                      periodicBoxVecY, 
    real4                                      periodicBoxVecZ
){
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < numExclusions; npair += blockDim.x*gridDim.x){
        int p1 = exclusionidx1[npair];
        int p2 = exclusionidx2[npair];
        int atom1 = indexAtom[p1];
        int atom2 = indexAtom[p2];
        real prm1 = charges[p1];
        real prm2 = charges[p2];
        real c1c2 = prm1 * prm2;
        real4 posq1 = posq[atom1];
        real4 posq2 = posq[atom2];
        real3 delta = make_real3(posq2.x - posq1.x, posq2.y - posq1.y, posq2.z - posq1.z);
        APPLY_PERIODIC_TO_DELTA(delta)
        
        real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        real invR = RSQRT(r2);
        real r = r2 * invR;
        real alphaR = EWALD_ALPHA * r;
        if (r < CUTOFF){
#ifdef USE_DOUBLE_PRECISION
            const real erfcAlphaR = erfc(alphaR);
#else
            const real t = RECIP(1.0f+0.3275911f*alphaR);
            const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
            energyBuffer[npair] -= ONE_4PI_EPS0 * c1c2 * invR;
            real dEdR = - ONE_4PI_EPS0 * c1c2 * invR * invR * invR;
            // dEdR = dEdR * (alphaR * EXP(- alphaR * alphaR) * TWO_OVER_SQRT_PI + erfcAlphaR);
            delta *= dEdR;

            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long) (-delta.x*0x100000000)));
            atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-delta.y*0x100000000)));
            atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-delta.z*0x100000000)));
            atomicAdd(&forceBuffers[atom2], static_cast<unsigned long long>((long long) (delta.x*0x100000000)));
            atomicAdd(&forceBuffers[atom2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (delta.y*0x100000000)));
            atomicAdd(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (delta.z*0x100000000)));
        }
    }
}

extern "C" __global__ void genIndexAtom(
    const int*     __restrict__    atomIndex,
    int*           __restrict__    indexAtom,
    int                            numParticles
){
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numParticles; atom += blockDim.x*gridDim.x){
        int index = atomIndex[atom];
        indexAtom[index] = atom;
    }
}

extern "C" __global__ void computeEwaldRecEner(
    mixed*                    __restrict__     energyBuffer, 
    const real4*              __restrict__     posq, 
    const real*               __restrict__     charges,
    const int*                __restrict__     atomIndex,
    real*                     __restrict__     cosSinSums,
    real4                                      periodicBoxSize,
    real4                                      invPeriodicBoxSize
){
    const unsigned int ksizex = 2*KMAX_X-1;
    const unsigned int ksizey = 2*KMAX_Y-1;
    const unsigned int ksizez = 2*KMAX_Z-1;
    const unsigned int totalK = ksizex*ksizey*ksizez;
    real3 reciprocalBoxSize = make_real3(2*M_PI*invPeriodicBoxSize.x, 2*M_PI*invPeriodicBoxSize.y, 2*M_PI*invPeriodicBoxSize.z);
    real reciprocalCoefficient = ONE_4PI_EPS0*4*M_PI*(invPeriodicBoxSize.x*invPeriodicBoxSize.y*invPeriodicBoxSize.z);
    unsigned int index = blockIdx.x*blockDim.x+threadIdx.x;
    mixed energy = 0;
    while (index < (KMAX_Y-1)*ksizez+KMAX_Z)
        index += blockDim.x*gridDim.x;
    while (index < totalK) {
        // Find the wave vector (kx, ky, kz) this index corresponds to.

        int rx = index/(ksizey*ksizez);
        int remainder = index - rx*ksizey*ksizez;
        int ry = remainder/ksizez;
        int rz = remainder - ry*ksizez - KMAX_Z + 1;
        ry += -KMAX_Y + 1;
        real kx = rx*reciprocalBoxSize.x;
        real ky = ry*reciprocalBoxSize.y;
        real kz = rz*reciprocalBoxSize.z;

        // Compute the sum for this wave vector.

        real cossum = 0;
        real sinsum = 0;
        for (int atom = 0; atom < NUM_ATOMS; atom++) {
            real4 apos = posq[atom];
            real costmp = COS(apos.x*kx + apos.y*ky + apos.z*kz);
            real sintmp = SIN(apos.x*kx + apos.y*ky + apos.z*kz);
            cossum += costmp * charges[atomIndex[atom]];
            sinsum += sintmp * charges[atomIndex[atom]];
        }
        cosSinSums[2*index] = cossum;
        cosSinSums[2*index+1] = sinsum;

        // Compute the contribution to the energy.

        real k2 = kx*kx + ky*ky + kz*kz;
        real ak = EXP(k2*EXP_COEFFICIENT) / k2;
        energy += reciprocalCoefficient*ak*(cossum*cossum + sinsum*sinsum);
        index += blockDim.x*gridDim.x;
    }
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
}

extern "C" __global__ void computeEwaldRecForce(
    unsigned long long*       __restrict__     forceBuffers, 
    const real4*              __restrict__     posq, 
    const real*               __restrict__     charges,
    const int*                __restrict__     atomIndex,
    const real*               __restrict__     cosSinSums, 
    real4                                      periodicBoxSize,
    real4                                      invPeriodicBoxSize
){
    unsigned int atom = blockIdx.x*blockDim.x+threadIdx.x;
    real3 reciprocalBoxSize = make_real3(2*M_PI*invPeriodicBoxSize.x, 2*M_PI*invPeriodicBoxSize.y, 2*M_PI*invPeriodicBoxSize.z);
    real reciprocalCoefficient = ONE_4PI_EPS0*4*M_PI*(invPeriodicBoxSize.x*invPeriodicBoxSize.y*invPeriodicBoxSize.z);
    while (atom < NUM_ATOMS) {
        real3 force = make_real3(0);
        real4 apos = posq[atom];

        // Loop over all wave vectors.

        int lowry = 0;
        int lowrz = 1;
        for (int rx = 0; rx < KMAX_X; rx++) {
            real kx = rx*reciprocalBoxSize.x;
            real phase1 = apos.x*kx;
            for (int ry = lowry; ry < KMAX_Y; ry++) {
                real ky = ry*reciprocalBoxSize.y;
                real phase2 = phase1 + apos.y*ky;
                for (int rz = lowrz; rz < KMAX_Z; rz++) {
                    real kz = rz*reciprocalBoxSize.z;

                    // Compute the force contribution of this wave vector.
                    int index = rx*(KMAX_Y*2-1)*(KMAX_Z*2-1) + (ry+KMAX_Y-1)*(KMAX_Z*2-1) + (rz+KMAX_Z-1);
                    real k2 = kx*kx + ky*ky + kz*kz;
                    real ak = EXP(k2*EXP_COEFFICIENT)/k2;
                    real phase3 = phase2 + apos.z*kz;
                    real2 structureFactor = make_real2(COS(phase3), SIN(phase3));
                    real cossum = cosSinSums[index*2];
                    real sinsum = cosSinSums[index*2+1];
                    real dEdR = 2*reciprocalCoefficient*ak*charges[atomIndex[atom]]*(cossum*structureFactor.y - sinsum*structureFactor.x);
                    force.x += dEdR*kx;
                    force.y += dEdR*ky;
                    force.z += dEdR*kz;
                    lowrz = 1 - KMAX_Z;
                }
                lowry = 1 - KMAX_Y;
            }
        }

        // Record the force on the atom.

        atomicAdd(&forceBuffers[atom], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[atom+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[atom+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
        atom += blockDim.x*gridDim.x;
    }
}