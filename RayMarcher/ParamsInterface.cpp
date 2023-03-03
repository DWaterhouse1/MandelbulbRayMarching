#include "ParamsInterface.hpp"

//std
#include <array>

//boost
#include <boost/numeric/conversion/cast.hpp>

namespace rmcuda
{
namespace
{
template<typename T, size_t arraySize>
class ComboBoxArrays
{
public:
	constexpr ComboBoxArrays(
		std::array<const char*, arraySize> names,
		std::array<const T, arraySize> values) :
		m_names{ names },
		m_values{ values }
	{}

	const T& operator [](size_t i) const { return m_values[i]; }
	const char* const* names() const { return m_names.data(); }
	size_t size() const { return m_names.size(); }

private:
	const std::array<const char*, arraySize> m_names;
	const std::array<const T, arraySize> m_values;
};
}

void ParamsInterface::onAttach()
{
	m_io = &ImGui::GetIO();
}

void ParamsInterface::onRender()
{
	if (m_visible)
	{
		showParamatersWindow();
	}
}

void ParamsInterface::tick(float deltaTime)
{
	if (m_exponentPlay)
	{
		m_exponent += (deltaTime * m_rate);

		if (m_exponent > m_exponentMax)
		{
			m_exponent += (m_exponentMin - m_exponentMax);
		}
	}
}

void ParamsInterface::showParamatersWindow()
{
	ImGui::Begin("Params Interface");

	framerateIndicator();

	ImGui::Separator();

	cameraControls();

	exponentControls();

	rayMarchingControls();

	shadingModeControls();

	ImGui::End();
}

void ParamsInterface::framerateIndicator()
{
	ImGui::Text("Framerate: "); ImGui::SameLine();
	if (m_io)
	{
		ImGui::Text((std::format("{:.2f}", m_io->Framerate) + " fps").c_str());
	}
	else
	{
		ImGui::Text("null");
	}
}

void ParamsInterface::cameraControls()
{
	if (ImGui::CollapsingHeader("Camera Controls"))
	{
		static float theta{ m_cameraCoords.theta * constants::r2d };
		static float phi{ m_cameraCoords.phi * constants::r2d };

		ImGui::SliderFloat("rho", &m_cameraCoords.rho, 0.5f, 5.0f);
		ImGui::SliderFloat("theta", &theta, -180.0f, 180.0f);
		ImGui::SliderFloat("phi", &phi, 0.0f, 180.0f);

		m_cameraCoords.theta = constants::d2r * theta;
		m_cameraCoords.phi = constants::d2r * phi;
	}
}

void ParamsInterface::exponentControls()
{
	if (ImGui::CollapsingHeader("Exponent"))
	{
		ImGui::Checkbox("Auto-play exponent ", &m_exponentPlay);

		if (m_exponentPlay)
		{
			ImGui::Text("exponent value");
			ImGui::Text(std::format("{:.2f}", m_exponent).c_str());
		}
		else
		{
			ImGui::SliderFloat("exponent value", &m_exponent, m_exponentMin, m_exponentMax, "%.2f");
		}
	}
}

void ParamsInterface::rayMarchingControls()
{
	static constexpr ComboBoxArrays<int, 5> msaaCount
	{
		{ "1", "2", "4", "8", "16" },
		{  1,   2,   4,   8,   16  }
	};

	static constexpr ComboBoxArrays<int, 4> resolutionScaling
	{
		{ "1/4", "1/3",	"1/2", "1"},
		{  4,			3,		 2,		  1 }
	};

	static int msaaCountIndex = 0;
	static int scalingIndex = 3;

	if (ImGui::CollapsingHeader("Ray Marching Parameters"))
	{
		if (ImGui::Combo(
			"Multisample Count",
			&msaaCountIndex,
			msaaCount.names(),
			boost::numeric_cast<int>(msaaCount.size())))
		{
			m_msaaCount = msaaCount[msaaCountIndex];
		}

		if (ImGui::Combo(
			"Scaling",
			&scalingIndex,
			resolutionScaling.names(),
			boost::numeric_cast<int>(resolutionScaling.size())))
		{
			m_resolutionScale = resolutionScaling[scalingIndex];
			m_scaleDirtyFlag = true;
		}
	}

}

void ParamsInterface::shadingModeControls()
{
}

} // namespace rmcuda