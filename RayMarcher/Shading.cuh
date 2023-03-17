#ifndef RAYMARCHER_SHADING_CUH
#define RAYMARCHER_SHADING_CUH

#include "Mandelbulb.cuh"

namespace rmcuda
{
namespace compute
{
/**
* These classes define strategies to be used in shading the mandelbulb renderer.
* Any compatible strategy must define the interface
*		__device__ float3 shade(float3 position, float exponent);
*		__device__ void setNumSteps(int steps);
*		--device__ void setMaxSteps(int steps);
* 
* The shade function defines the RGB value that will be contributed by a given
* ray intersection to the pixel colour, as the float3 it returns.
*/

/**
* This strategy shades based on a very simple lambert model relative to a static
* light at a hardcoded position of (2.0, -5.0, 3.0). The set_Steps functions are
* left empty implemented, since these parameters are not used. The constructor
* parameter diffuseColour is used as the base object colour against which the
* cosine distributed shading intensity will be applied.
*/
class DiffuseStrategy
{
public:
	__host__ DiffuseStrategy(float3 diffuseColour)
		: m_diffuseColour{ diffuseColour }
	{}

	__device__ float3 shade(float3 position, float exponent) const
	{
		float3 normal = mandelbulbNormal(position, exponent);

		float3 lightPosition = make_float3(2.0f, -5.0f, 3.0f);

		float3 lightDirection = normalize(position - lightPosition);

		float intensity = max(0.0f, dot(normal, lightDirection));

		return m_diffuseColour * intensity;
	}

	__device__ void setNumSteps(int steps) {}
	__device__ void setMaxSteps(int steps) {}

private:
	const float3 m_diffuseColour;
};

/**
* This strategy shades simply based on the calculated normal direction. The
* normal vector n is mapped to an RGB colour space (0, 1)^3 by 0.5 * n + 0.5
* The set_Steps functions are left empty implemented, since these parameters
* are not used.
*/
class NormalStrategy
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
* This strategy shades by interpolating between two colours based on the number
* of steps taken to intersect. The ratio between the intersection step and the
* maximum step limit is taken and then scaled by an exponent of 0.3f. This
* resultant value is used to interpolate between the colours supplied in the
* constructor as nearColour and farColour.
*/
class StepwiseStrategy
{
public:
	__host__ StepwiseStrategy(float3 nearColour, float3 farColour)
		: m_nearColour{ nearColour },
			m_farColour{ farColour }
	{}

	__device__ float3 shade(float3 position, float exponent) const
	{
		const float stepRatio =
			static_cast<float>(m_nearSteps) /
			static_cast<float>(m_farSteps);

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
	const float3 m_nearColour;
	const float3 m_farColour;

	int m_nearSteps = 0;
	int m_farSteps = 0;
};

} // namespace compute
} // namespace rmcuda

#endif // RAYMARCHER_SHADING_CUH
