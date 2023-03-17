#include "RayMarchCompute.cuh"

#define USE_GPU_TRIG

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "Shading.cuh"
#include "Mandelbulb.cuh"

// nvcc will compile qualified namespaces, but it breaks intellisense
namespace rmcuda
{
namespace compute
{
/**
* Marches the supplied ray and returns the colour from any intersection. Shading
* is defined by the ShadingStrategy.
* 
* @tparam ShadingStrategy The type of the ShadingStrategy object to use.
* 
* @param ray Ray to march along.
* @param exponent Exponent value to use in mandelbulb calculations.
* @param shadingStrategy Concrete shading strategy object to be used to calculate
*		colour values.
* 
* @return The colour this marched ray should contribute to the pixel colour.
*/
template<typename ShadingStrategy>
__device__ float3 march(Ray ray, float exponent, ShadingStrategy shadingStrategy)
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
		;
		float stepDistance = mandelbulbDistance(currentPosition, exponent);

		if (stepDistance < minDistance)
		{
			// hit

			shadingStrategy.setNumSteps(step);

			return shadingStrategy.shade(currentPosition, exponent);
		}

		if (totalDistance > maxDistance) break;

		totalDistance += stepDistance;
	}

	// didn't hit
	return make_float3(0.05f, 0.05f, 0.05f);
}

/**
* Entry point for the ray marching algorithm used to calculate the mandelbulb
*		image.
* 
* @tparam Strategy The shading strategy to use.
* 
* @param surface Handle to the CUDA surface object to be used for drawing the
*		mandelbulb
* @param pixelDim Specifies in the x and y elements the image extent.
* @param camera Defines the camera location and orientation.
* @param exponent The exponent value to be used in mandelbulb calculations.
* @param numSamples The number of samples to take for a given pixel.
* @param shadingStrategy The strategy object used to calulate colour.
*/
template<typename Strategy>
__global__ void calculateMandelbulb(
	cudaSurfaceObject_t surface,
	dim3 pixelDim,
	Camera camera,
	float exponent,
	int numSamples,
	Strategy shadingStrategy)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= pixelDim.x || y >= pixelDim.y) return;

	//TODO seed this with a random value
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

		color += march<Strategy>(ray, exponent, shadingStrategy);
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

void rayMarchDiffuseColour(
	cudaSurfaceObject_t surface,
	dim3 texDim,
	Camera camera,
	float exponent,
	int numSamples,
	float3 colour)
{
	DiffuseStrategy shadingStrategy{ colour };

	dim3 thread(16, 16);
	dim3 block(texDim.x / thread.x, texDim.y / thread.y);
	calculateMandelbulb<<<block, thread>>>(
		surface,
		texDim,
		camera,
		exponent,
		numSamples,
		shadingStrategy);
}

void rayMarchNormalColour(
	cudaSurfaceObject_t surface,
	dim3 texDim,
	Camera camera,
	float exponent,
	int numSamples)
{
	NormalStrategy shadingStrategy{};

	dim3 thread(16, 16);
	dim3 block(texDim.x / thread.x, texDim.y / thread.y);
	calculateMandelbulb<<<block, thread>>>(
		surface,
		texDim,
		camera,
		exponent,
		numSamples,
		shadingStrategy);
}

void rayMarchStepwiseColour(
	cudaSurfaceObject_t surface,
	dim3 texDim,
	Camera camera,
	float exponent,
	int numSamples,
	float3 lowColour,
	float3 highColour)
{
	StepwiseStrategy shadingStrategy{ lowColour, highColour };

	dim3 thread(16, 16);
	dim3 block(texDim.x / thread.x, texDim.y / thread.y);
	calculateMandelbulb<<<block, thread>>>(
		surface,
		texDim,
		camera,
		exponent,
		numSamples,
		shadingStrategy);
}

} // namespace rmcuda
} // namespace compute