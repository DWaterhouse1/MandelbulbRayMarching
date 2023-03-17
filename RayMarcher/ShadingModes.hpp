#ifndef RAYMARCHER_SHADINGMODES_HPP
#define RAYMARCHER_SHADINGMODES_HPP

namespace rmcuda
{
/**
* Enumerates the possible shading modes.
*/
enum class ShadingMode
{
	kDiffuseLight,
	kNormalColor,
	kStepwiseShaded,
};

} // namespace rmcuda

#endif // RAYMARCHER_SHADINGMODES_HPP