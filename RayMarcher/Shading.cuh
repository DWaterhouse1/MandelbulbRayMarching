#ifndef RAYMARCHER_SHADING_CUH
#define RAYMARCHER_SHADING_CUH

namespace rmcuda
{
namespace compute
{
extern __device__ float3 mandelbulbNormal(float3 pos, float exponent);

struct Diffuse
{
	__device__ static float3 shade(float3 position, float exponent, float3 inColour)
	{
		float3 normal = mandelbulbNormal(position, exponent);

		float3 lightPosition = make_float3(2.0, -5.0, 3.0);

		float3 lightDirection = normalize(position - lightPosition);

		float intensity = max(0.0f, dot(normal, lightDirection));

		return inColour * intensity;
	}
};

struct Normal
{
	__device__ static float3 shade(float3 position, float exponent, float3 inColour)
	{
		return 0.5f * mandelbulbNormal(position, exponent) + 0.5f;
	}
};

struct Depth
{
	__device__ static float3 shade(float3 position, float exponent, float3 inColour)
	{

	}
};

template<typename ShadingPolicy>
__device__ float3 testMarch(Ray ray, float exponent, float3 inColour)
{
	float totalDistance = 0.0f;
	const float minDistance = 0.0001f;
	const float maxDistance = 100.0f;
	const int maxSteps = 2000;

	const float3 sphereCenter = make_float3(0.0f);

	for (int i = 0; i < maxSteps; ++i)
	{
		float3 currentPosition = ray.origin + (totalDistance * ray.direction);

		//float stepDistance = sphereDistance(currentPosition, sphereCenter, 1.0f);
		float stepDistance = mandelbulbDistance(currentPosition, exponent);

		if (stepDistance < minDistance)
		{
			// hit
			return ShadingPolicy::shade(currentPosition, exponent, inColour);
		}

		if (totalDistance > maxDistance) break;

		totalDistance += stepDistance;
	}

	// didn't hit
	return make_float3(0.05f, 0.05f, 0.05f);
}

} // namespace compute
} // namespace rmcuda

#endif // RAYMARCHER_SHADING_CUH
