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

class ParamsInterface : public wrndr::InterfaceLayer
{
public:
	ParamsInterface(SphericalCoord initialCameraPos = SphericalCoord())
		: m_cameraCoords{ initialCameraPos }
	{}

	virtual void onAttach() override;

	virtual void onRender() override;

	virtual void tick(float deltaTime) override;

	void setExponentScrollRate(float rate) { m_rate = rate; }

	void setExponent(float exp) { }

	[[nodiscard]] float getExponent() const { return m_exponent; }

	[[nodiscard]] float getCameraRho() const { return m_cameraCoords.rho; }

	[[nodiscard]] float getCameraTheta() const { return m_cameraCoords.theta; }

	[[nodiscard]] float getCameraPhi() const { return m_cameraCoords.phi; }

	[[nodiscard]] SphericalCoord getCameraCoords() const { return m_cameraCoords; }

	[[nodiscard]] int getSampleCount() const { return m_msaaCount; }

	[[nodiscard]] int getResolution() const { return m_resolution; }

	[[nodiscard]] bool resScaleDirty() const { return m_scaleDirtyFlag; }

	[[nodiscard]] int getResolutionScale() const { return m_resolutionScale; }

	[[nodiscard]] ShadingMode getShadingMode() const { return m_activeShadingMode; }

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
	int m_resolution = 512;
	int m_resolutionScale = 1;
	bool m_scaleDirtyFlag = false;

	ImGuiIO* m_io = nullptr;

	SphericalCoord m_cameraCoords{};

	ShadingMode m_activeShadingMode = ShadingMode::kDiffuseLight;
};

} // namespace rmcuda

#endif // RAYMARCHER_PARAMSINTERFACE_HPP