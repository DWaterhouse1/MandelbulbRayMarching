#ifndef RAYMARCHER_MANDELBULB_CUH
#define RAYMARCHER_MANDELBULB_CUH

#include "cuda_runtime.h"

namespace rmcuda
{
namespace compute
{
__device__ extern float mandelbulbDistance(float3 position, float exponent);

__device__ extern float3 mandelbulbNormal(float3 pos, float exponent);

// not used currently

__device__ float sphereDistance(float3 position, float3 center, float radius);
__device__ float3 sphereNormal(float3 pos, float3 center, float radius);

} // namespace compute
} // namespace rmcuda

#endif // RAYMARCHER_MANDELBULB_CUH