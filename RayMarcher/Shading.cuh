#ifndef RAYMARCHER_SHADING_CUH
#define RAYMARCHER_SHADING_CUH

namespace rmcuda
{
namespace compute
{
extern __device__ float3 mandelbulbNormal(float3 pos, float exponent);

struct Diffuse
{
	__device__ static float3 shade(float3 position, float exponent, float3 inColour, float stepRatio)
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
	__device__ static float3 shade(float3 position, float exponent, float3 inColour, float stepRatio)
	{
		return 0.5f * mandelbulbNormal(position, exponent) + 0.5f;
	}
};

struct Stepwise
{
	__device__ static float3 shade(float3 position, float exponent, float3 inColour, float stepRatio)
	{
		const float correction = 0.3f;

		const float3 initialColour = make_float3(1.0f);
		const float3 finalColour = make_float3(0.0f);

		/* The mandelbulb is a very complicated shape, and so a very high value of
		*  max steps is required in order to capture all the detail. However,
		*  relatively few rays will actually use all these steps to complete. So the
		*  stepRatio will be very skewed, and so a correction is needed to pull out
		*  the aesthetically interesting features of the object.
		*/
		float mix = pow(stepRatio, correction);

		return mix * initialColour + (1 - mix) * finalColour;
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

	for (int step = 0; step < maxSteps; ++step)
	{
		float3 currentPosition = ray.origin + (totalDistance * ray.direction);

		//float stepDistance = sphereDistance(currentPosition, sphereCenter, 1.0f);
		float stepDistance = mandelbulbDistance(currentPosition, exponent);

		if (stepDistance < minDistance)
		{
			// hit
			return ShadingPolicy::shade(currentPosition, exponent, inColour, (float)step / float(maxSteps));
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
