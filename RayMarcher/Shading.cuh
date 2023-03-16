#ifndef RAYMARCHER_SHADING_CUH
#define RAYMARCHER_SHADING_CUH

namespace rmcuda
{
namespace compute
{
extern __device__ float3 mandelbulbNormal(float3 pos, float exponent);

/**
* Interface for the different shading modes.
* 
* @tparam Derived The derived class for the CRTP pattern.
*/
template <typename Derived>
class ShadingStrategy
{
public:
	__device__ float3 colour(float3 position, float exponent) const
	{
		Derived& derived = static_cast<Derived&>(*this);
		return derived.shade(position, exponent);
		//return make_float3(0.4f, 0.3f, 0.8f);
	}

	__device__ void setNumSteps(int steps)
	{
		Derived& derived = static_cast<Derived&>(*this);
		derived.setNumSteps(steps);
	}

	__device__ void setMaxSteps(int steps)
	{
		Derived& derived = static_cast<Derived&>(*this);
		derived.setMaxSteps(steps);
	}
};

/**
* 
*/
class DiffuseShading : public ShadingStrategy<DiffuseShading>
{
public:
	__host__ DiffuseShading(float3 diffuseColour)
		: m_diffuseColour{ diffuseColour }
	{}

	__device__ float3 shade(float3 position, float exponent) const
	{
		float3 normal = mandelbulbNormal(position, exponent);

		float3 lightPosition = make_float3(2.0, -5.0, 3.0);

		float3 lightDirection = normalize(position - lightPosition);

		float intensity = max(0.0f, dot(normal, lightDirection));

		return m_diffuseColour * intensity;
	}

	__device__ void setNumSteps(int steps) {}
	__device__ void setMaxSteps(int steps) {}

private:
	float3 m_diffuseColour;
};

/**
* 
*/
class NormalShading : public ShadingStrategy<NormalShading>
{
public:
	__device__ float3 shade(float3 position, float exponent) const
	{
		return 0.5f * mandelbulbNormal(position, exponent) + 0.5f;
	}

	__device__ void setNumSteps(int steps) {}
	__device__ void setMaxSteps(int steps) {}
};

/**
* 
*/
class StepwiseShading : public ShadingStrategy<StepwiseShading>
{
public:
	__host__ StepwiseShading(float3 nearColour, float3 farColour)
		: m_nearColour{ nearColour },
			m_farColour{ farColour }
	{}

	__device__ float3 shade(float3 position, float exponent) const
	{
		float stepRatio = static_cast<float>(m_nearSteps) / static_cast<float>(m_farSteps);

		/* The mandelbulb is a very complicated shape, and so a very high value of
		*  max steps is required in order to capture all the detail. However,
		*  relatively few rays will actually use all these steps to complete. So the
		*  stepRatio will be very skewed, and a correction is needed to pull out
		*  the aesthetically interesting features of the object.
		*/
		static constexpr float correction = 0.3f;

		float mix = pow(stepRatio, correction);

		return mix * m_nearColour + (1 - mix) * m_farColour;
	}

	__device__ void setNumSteps(int steps) { m_nearSteps = steps; }
	__device__ void setMaxSteps(int steps) { m_farSteps = steps; }

private:
	float3 m_nearColour;
	float3 m_farColour;

	int m_nearSteps = 0;
	int m_farSteps = 0;
};

template<typename Mode>
__device__ float3 testMarch(Ray ray, float exponent, ShadingStrategy<Mode> shadingStrategy)
{
	float totalDistance = 0.0f;
	const float minDistance = 0.0001f;
	const float maxDistance = 100.0f;
	const int maxSteps = 2000;

	shadingStrategy.setMaxSteps(maxSteps);

	const float3 sphereCenter = make_float3(0.0f);

	for (int step = 0; step < maxSteps; ++step)
	{
		float3 currentPosition = ray.origin + (totalDistance * ray.direction);

		//float stepDistance = sphereDistance(currentPosition, sphereCenter, 1.0f);
		float stepDistance = mandelbulbDistance(currentPosition, exponent);

		if (stepDistance < minDistance)
		{
			// hit

			shadingStrategy.setNumSteps(step);

			return shadingStrategy.colour(currentPosition, exponent);
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
