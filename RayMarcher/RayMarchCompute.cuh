#ifndef RAYMARCHER_RAYMARCHCOMPUTE_CUH
#define RAYMARCHER_RAYMARCHCOMPUTE_CUH

#include "vendor/helper_math.h"

namespace rmcuda
{

/**
* Structure which holds camera position and orientation in terms of a set of
* camera local orthonormal basis vectors, along with field of view values stored
* as the reciprocal of the tangent of half of the field of view in radians.
*/
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
/**
* Struct defining a ray in terms of position and direction.
*/
struct Ray
{
	float3 origin;
	float3 direction;
};

/**
* Launches a ray marching computation on the supplied image agains a mandelbulb
*		SDF. Shades based on a very basic lambertian diffuse colour model.
*
* @param surface Handle to the CUDA surface on which to draw the image.
* @param texDim Defines the image extent.
* @param camera Defines the camera position and orientation.
* @param exponent Defines the exponent value to be uesd in the manedlbulb
*		calculation.
* @param numSamples Defines the number of samples to take per pixel.
* @param colour Base colour to use for the shading model.
*/
extern void rayMarchDiffuseColour(
	cudaSurfaceObject_t surface,
	dim3 texDim,
	Camera camera,
	float exponent,
	int numSamples,
	float3 colour);

/**
* Launches a ray marching computation on the supplied image agains a mandelbulb
*		SDF. Shades based on calculated normal values transformed to RGB.
*
* @param surface Handle to the CUDA surface on which to draw the image.
* @param texDim Defines the image extent.
* @param camera Defines the camera position and orientation.
* @param exponent Defines the exponent value to be uesd in the manedlbulb
*		calculation.
* @param numSamples Defines the number of samples to take per pixel.
*/
extern void rayMarchNormalColour(
	cudaSurfaceObject_t surface,
	dim3 texDim,
	Camera camera,
	float exponent,
	int numSamples);

/**
* Launches a ray marching computation on the supplied image agains a mandelbulb
*		SDF. Shades by interpolating between supplied colours, based on the number
*		of steps taken by the ray marching.
*
* @param surface Handle to the CUDA surface on which to draw the image.
* @param texDim Defines the image extent.
* @param camera Defines the camera position and orientation.
* @param exponent Defines the exponent value to be uesd in the manedlbulb
*		calculation.
* @param numSamples Defines the number of samples to take per pixel.
* @param lowColour Colour used for lower number of steps taken.
* @param highColour Colur used for high number of steps taken.
*/
extern void rayMarchStepwiseColour(
	cudaSurfaceObject_t surface,
	dim3 texDim,
	Camera camera,
	float exponent,
	int numSamples,
	float3 lowColour,
	float3 highColour);

} // namespace compute
} // namespace rmcuda

#endif // RAYMARCHER_RAYMARCHCOMPUTE_CUH