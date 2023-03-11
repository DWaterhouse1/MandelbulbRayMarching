#ifndef RAYMARCHER_SHADINGMODES_HPP
#define RAYMARCHER_SHADINGMODES_HPP

namespace rmcuda
{
/**
* 
*/
enum class ShadingMode
{
	kDiffuseLight,
	kNormalColor,
	kStepwiseShaded,
};

} // namespace rmcuda

#endif // RAYMARCHER_SHADINGMODES_HPP