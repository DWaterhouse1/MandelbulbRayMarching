#ifndef RAYMARCHER_MANDELBULB_CUH
#define RAYMARCHER_MANDELBULB_CUH

#include "cuda_runtime.h"

namespace rmcuda
{
namespace compute
{
/**
* Calculates the value of the mandelbulb signed distance field at a point.
* 
* @param position 3D position at which to evaluate the SDF.
* @param exponent Exponent value to use in the SDF calculation.
* 
* @return Value of the SDF at position.
*/
__device__ extern float mandelbulbDistance(float3 position, float exponent);

/**
* Calculates the gradient of the mandelbulb signed distance field at a point.
* 
* @param pos 3D position at which to evaluate the gradient.
* @param exponent Exponent value to use in the gradient calculation.
* 
* @return Vector value of gradient at position.
*/
__device__ extern float3 mandelbulbNormal(float3 pos, float exponent);

// not used, left for illustrative purposes
__device__ float sphereDistance(float3 position, float3 center, float radius);

// not used
__device__ float3 sphereNormal(float3 pos, float3 center, float radius);

} // namespace compute
} // namespace rmcuda

#endif // RAYMARCHER_MANDELBULB_CUH