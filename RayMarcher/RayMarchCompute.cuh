#ifndef RAYMARCHER_RAYMARCHCOMPUTE_CUH
#define RAYMARCHER_RAYMARCHCOMPUTE_CUH

#include "vendor/helper_math.h"

namespace rmcuda
{
struct Camera
{
	float3 pos;
	float3 dir;
	float3 up;
	float3 right;
	float invhalffov;

	void lookAt(float3 target)
	{
		const float3 globalUp = make_float3(0.0f, 0.0f, 1.0f);

		dir = normalize(target - pos);
		right = normalize(cross(globalUp, dir));
		up = cross(dir, right);
	}
};

namespace compute
{
struct Ray
{
	float3 origin;
	float3 direction;
};

extern void basicRayMarching(
	cudaSurfaceObject_t surface,
	dim3 texDim,
	Camera camera,
	float exponent,
	int numSamples);

extern void rayMarchDiffuseColour(
	cudaSurfaceObject_t surface,
	dim3 texDim,
	Camera camera,
	float exponent,
	int numSamples,
	float3 colour);

extern void rayMarchNormalColour(
	cudaSurfaceObject_t surface,
	dim3 texDim,
	Camera camera,
	float exponent,
	int numSamples);

extern void rayMarchStepwiseColour(
	cudaSurfaceObject_t surface,
	dim3 texDim,
	Camera camera,
	float exponent,
	int numSamples);

} // namespace compute
} // namespace rmcuda

#endif // RAYMARCHER_RAYMARCHCOMPUTE_CUH