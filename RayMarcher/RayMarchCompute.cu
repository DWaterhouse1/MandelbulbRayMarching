#include "RayMarchCompute.cuh"

#define USE_GPU_TRIG

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

// nvcc will compile qualified namespaces, but it breaks intellisense
namespace rmcuda
{
namespace compute
{
void basicRayMarching(cudaSurfaceObject_t surface, dim3 texDim, Camera camera, float exponent, int numSamples)
{
	dim3 thread(16, 16);
	dim3 block(texDim.x / thread.x, texDim.y / thread.y);
	rayMarch<<<block, thread>>>(surface, texDim, camera, exponent, numSamples);
}

// TODO expand to alternate aspect ratios/resolution scalings
__global__ void rayMarch(cudaSurfaceObject_t surface, dim3 pixelDim, Camera camera, float exponent, int numSamples)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= pixelDim.x || y >= pixelDim.y) return;

	curandState state;
	curand_init(1729, 0, 0, &state);

	float3 color = make_float3(0.0f);

	for (int i = 0; i < numSamples; ++i)
	{
		// mapping (0, width) X (0, height) -> (-1, 1)^2
		float xNorm = (float(x + curand_uniform(&state)) / float(pixelDim.x) - 0.5f) * 2.0f;
		float yNorm = (float(y + curand_uniform(&state)) / float(pixelDim.y) - 0.5f) * 2.0f;

		Ray ray =
		{
			camera.pos,
			normalize(
				camera.right * xNorm +
				camera.up * yNorm +
				camera.dir * camera.invhalffov)
		};

		color += march(ray, exponent);
	}

	color /= numSamples;

	if (x < pixelDim.x && y < pixelDim.y)
	{
		uchar4 data = make_uchar4(
			color.x * 255,
			color.y * 255,
			color.z * 255,
			255);
		surf2Dwrite(data, surface, x * sizeof(uchar4), y);
	}
}

__device__ float3 march(Ray ray, float exponent)
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
#if 1
			float3 normal = mandelbulbNormal(currentPosition, exponent);

			float3 lightPosition = make_float3(2.0, -5.0, 3.0);

			float3 lightDirection = normalize(currentPosition - lightPosition);

			float intensity = max(0.0f, dot(normal, lightDirection));

			return make_float3(1.0f, 1.0f, 1.0f) * intensity;
			//return make_float3(0.3f, 0.5f, 0.8f) * intensity;
#else
			return 0.5f * mandelbulbNormal(currentPosition, exponent) + 0.5f;
#endif
		}

		if (totalDistance > maxDistance) break;

		totalDistance += stepDistance;
	}

	// didn't hit
	return make_float3(0.05f, 0.05f, 0.05f);
}

__device__ float mandelbulbDistance(float3 position, float exponent)
{
	float3 z = position;

	const int maxItt = 20;
	//const float exponent = 9.0f;

	//int steps = 0;

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

__device__ float3 mandelbulbNormal(float3 pos, float exponent)
{
	const float peturb = 1e4;

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

__device__ float sphereDistance(float3 position, float3 center, float radius)
{
	return length(position - center) - radius;
}
} // namespace rmcuda
} // namespace compute