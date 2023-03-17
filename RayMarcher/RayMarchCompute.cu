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
// TODO respect differing aspect ratios
template<typename Strategy>
__global__ void rayMarch(
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
	rayMarch<<<block, thread>>>(
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
	rayMarch<<<block, thread>>>(
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
	rayMarch<<<block, thread>>>(
		surface,
		texDim,
		camera,
		exponent,
		numSamples,
		shadingStrategy);
}

} // namespace rmcuda
} // namespace compute