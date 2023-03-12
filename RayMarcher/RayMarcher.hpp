#ifndef RAYMARCHER_RAYMARCHER_HPP
#define RAYMARCHER_RAYMARCHER_HPP

#include "CudaTexture.hpp"
#include "ParamsInterface.hpp"

// wrenderer
#include <Window.hpp>
#include <Buffer.hpp>
#include <Shader.hpp>
#include <Program.hpp>
#include <VertexArray.hpp>
#include <UI.hpp>

// std
#include <vector>
#include <memory>

// boost
#include <boost/optional.hpp>

namespace rmcuda
{
class RayMarcher
{
public:
	RayMarcher(const int width = 512, const int height = 512);

	void run();

private:
	void checkResize();
	void updateMultisamples();

	wrndr::Window m_window;
	wrndr::Buffer<float> m_vertexBuffer;
	wrndr::Buffer<unsigned int> m_indexBuffer;
	wrndr::Shader m_vertexShader;
	wrndr::Shader m_fragmentShader;
	wrndr::Program m_shaderProgram;
	wrndr::VertexArray m_vertexArray;
	wrndr::UI m_interfaceContext;

	// note: nvcc compiler has issues with std::optional
	boost::optional<CudaInteropTexture> m_texture;
	std::shared_ptr<ParamsInterface> m_paramsInterface;
	unsigned int m_resizeFuncHandle = 0;

	bool m_resizeDirtyFlag = false;
	int m_resolutionScaling = 1;
	int m_resizeWidth;
	int m_resizeHeight;

	int m_numSamples = 1;
};

} // namespace rmcuda

#endif // RAYMARCHER_RAYMARCHER_HPP