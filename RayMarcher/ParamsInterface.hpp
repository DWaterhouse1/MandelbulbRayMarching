#ifndef RAYMARCHER_PARAMSINTERFACE_HPP
#define RAYMARCHER_PARAMSINTERFACE_HPP

#include "ShadingModes.hpp"

// wrenderer
#include "UILayer.hpp"

//std
#include <string>
#include <memory>
#include <format>

namespace rmcuda
{
namespace constants
{
static constexpr float PI = 3.141592f;
static constexpr float d2r = PI / 180.0f;
static constexpr float r2d = 180.0f / PI;
} // namespace constants

/**
* Tuple of float values representing spherical coordinates in radians.
*/
struct SphericalCoord
{
	float rho = 1.5f;
	float theta = 0.0f;
	float phi = 90.0f * constants::d2r;

	[[nodiscard]] float x() const { return rho * sinf(phi) * cosf(theta); }

	[[nodiscard]] float y() const { return rho * sinf(phi) * sinf(theta); }

	[[nodiscard]] float z() const { return rho * cosf(phi); }

	void setWithCartesian(float x, float y, float z)
	{
		rho = sqrtf((x * x) + (y * y) + (z * z));
		theta = atanf(y / x);
		phi = acosf(z / rho);
	}
};

/**
* Tuple of floats representing an RGB colour value.
*/
struct Colour
{
	float r = 1.0f;
	float g = 1.0f;
	float b = 1.0f;
};

/**
* User interface object for managing camera position and ray marching parameters.
*/
class ParamsInterface : public wrndr::InterfaceLayer
{
public:
	ParamsInterface(SphericalCoord initialCameraPos = SphericalCoord())
		: m_cameraCoords{ initialCameraPos }
	{}

	virtual void onAttach() override;

	virtual void onRender() override;

	virtual void tick(float deltaTime) override;

	/**
	* Sets the rate at which the exponent should advance in some unit time.
	* 
	* @param rate Rate to advance.
	*/
	void setExponentScrollRate(float rate) { m_rate = rate; }

	/**
	* Gets the current value of the mandelbulb exponent.
	* 
	* @return Exponent value.
	*/
	[[nodiscard]] float getExponent() const { return m_exponent; }

	/**
	* Gets the Rho value for the current camera position.
	* 
	* @return Rho value.
	*/
	[[nodiscard]] float getCameraRho() const { return m_cameraCoords.rho; }

	/**
	* Gets the Theta value for the current camera position.
	*
	* @return Theta value.
	*/
	[[nodiscard]] float getCameraTheta() const { return m_cameraCoords.theta; }

	/**
	* Gets the Phi value for the current camera position.
	*
	* @return Phi value.
	*/
	[[nodiscard]] float getCameraPhi() const { return m_cameraCoords.phi; }

	/**
	* Gets the current camera position in spherical coordinates.
	* 
	* @return Camera coordinates.
	*/
	[[nodiscard]] SphericalCoord getCameraCoords() const { return m_cameraCoords; }

	/**
	* Gets the current value for the multisample count.
	* 
	* @return Multisample count
	*/
	[[nodiscard]] int getSampleCount() const { return m_msaaCount; }

	/**
	* Gets the value of the dirty flag indicating the resolution needs to be updated.
	* 
	* @return True if resolution scale is dirty.
	*/
	[[nodiscard]] bool resScaleDirty() const { return m_scaleDirtyFlag; }

	/**
	* Gets the current value of the resolution scaling factor.
	* 
	* @return Resolution scaling factor.
	*/
	[[nodiscard]] int getResolutionScale() const { return m_resolutionScale; }

	/**
	* Gets the value of the current shading mode.
	* 
	* @return shading mode.
	*/
	[[nodiscard]] ShadingMode getShadingMode() const { return m_activeShadingMode; }

	/**
	* Gets the current diffuse shading colour value.
	* 
	* @return Colour.
	*/
	[[nodiscard]] Colour getDiffuseColour() const { return m_diffuseColour; }

	/**
	* Gets the current colour value for low step numbers.
	* 
	* @return Colour
	*/
	[[nodiscard]] Colour getLowStepColour() const { return m_lowColour; }

	/**
	* Gets the current colour value for high step numbers.
	* 
	* @return Colour.
	*/
	[[nodiscard]] Colour getHighStepColour() const { return m_highColour; }

	/**
	* Resets the resolution scaling dirty flag, to indicate that resolution has
	* been successfully updated.
	*/
	void resetResScaleDirty() { m_scaleDirtyFlag = false; }

private:
	void showParamatersWindow();
	void framerateIndicator();
	void cameraControls();
	void exponentControls();
	void rayMarchingControls();
	void shadingModeControls();

	bool m_visible = true;

	const float m_exponentMax = 20.0f;
	const float m_exponentMin = 2.0f;
	float m_exponent = 4.0f;
	float m_rate = 1.0f;
	bool m_exponentPlay = true;

	int m_msaaCount = 1;
	int m_resolutionScale = 1;
	bool m_scaleDirtyFlag = false;

	ImGuiIO* m_io = nullptr;

	SphericalCoord m_cameraCoords{};

	ShadingMode m_activeShadingMode = ShadingMode::kDiffuseLight;

	Colour m_diffuseColour = { 1.0f, 1.0f, 1.0f };
	Colour m_lowColour = { 1.0f, 1.0f, 1.0f };
	Colour m_highColour = { 0.0f, 0.0f, 0.0f };
};

} // namespace rmcuda

#endif // RAYMARCHER_PARAMSINTERFACE_HPP