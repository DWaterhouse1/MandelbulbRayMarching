#include "Mandelbulb.cuh"

#include "vendor/helper_math.h"

namespace rmcuda
{
namespace compute
{
__device__ extern float mandelbulbDistance(float3 position, float exponent)
{
	float3 z = position;

	static constexpr int maxItt = 20;

	float dr = 1.0f;
	float r = 0.0f;

	for (int i = 0; i < maxItt; ++i)
	{
		//steps = i;
		r = length(z);
		if (r > 4.0f) break;

		// convert to spherical coordinates
		float theta = acos(z.z / r);
		float phi = atan(z.y / z.x);
		dr = pow(r, exponent - 1.0f) * exponent * dr + 1.0f;

		// scale and rotate
		float zr = pow(r, exponent);
		theta *= exponent;
		phi *= exponent;

		// convert back to cartesian
		z = zr * make_float3(
			sin(theta) * cos(phi),
			sin(theta) * sin(phi),
			cos(theta));

		z += position;
	}

	return 0.5f * log(r) * r / dr;
}

// TODO add a generalised normal implementation that can caluluate normals using
// a supplied functor of some kind

__device__ extern float3 mandelbulbNormal(float3 pos, float exponent)
{
	static constexpr float peturb = 1e-2;

	float fx =
		mandelbulbDistance(make_float3(pos.x + peturb, pos.y, pos.z), exponent) -
		mandelbulbDistance(make_float3(pos.x - peturb, pos.y, pos.z), exponent);
	float fy =
		mandelbulbDistance(make_float3(pos.x, pos.y + peturb, pos.z), exponent) -
		mandelbulbDistance(make_float3(pos.x, pos.y - peturb, pos.z), exponent);
	float fz =
		mandelbulbDistance(make_float3(pos.x, pos.y, pos.z + peturb), exponent) -
		mandelbulbDistance(make_float3(pos.x, pos.y, pos.z - peturb), exponent);

	return normalize(make_float3(fx, fy, fz));
}


// sphere functions aren't used for now

__device__ float sphereDistance(float3 position, float3 center, float radius)
{
	return length(position - center) - radius;
}

__device__ float3 sphereNormal(float3 pos, float3 center, float radius)
{
	const float peturb = 1e4;

	float fx =
		sphereDistance(make_float3(pos.x + peturb, pos.y, pos.z), center, radius) -
		sphereDistance(make_float3(pos.x - peturb, pos.y, pos.z), center, radius);
	float fy =
		sphereDistance(make_float3(pos.x, pos.y + peturb, pos.z), center, radius) -
		sphereDistance(make_float3(pos.x, pos.y - peturb, pos.z), center, radius);
	float fz =
		sphereDistance(make_float3(pos.x, pos.y, pos.z + peturb), center, radius) -
		sphereDistance(make_float3(pos.x, pos.y, pos.z - peturb), center, radius);

	return normalize(make_float3(fx, fy, fz));
}

} // namespace compute
} // namespace rmcuda