#include "RayMarcher.hpp"

#include "Timer.hpp"
#include "RayMarchCompute.cuh"

// wrenderer
#include <ShaderCode.hpp>
#include <Renderer.hpp>

// std
#include <iostream>

namespace rmcuda
{
RayMarcher::RayMarcher(const int width, const int height) :
  m_window{ "Mandelbulb Ray Marcher", width, height },
  m_indexBuffer{},
  m_vertexBuffer{},
  m_vertexShader{
    wrndr::ShaderType::kVertex,
    std::string(wrndr::constants::vertexShaderSource) },
  m_fragmentShader{
      wrndr::ShaderType::kFragment,
      std::string(wrndr::constants::fragmentShaderSource) },
  m_shaderProgram{ m_vertexShader, m_fragmentShader },
  m_vertexArray{},
  m_interfaceContext{ m_window },
  m_resizeWidth{ width },
  m_resizeHeight{ height },
  m_paramsInterface{ nullptr }
{
  initBuffers();
  
  initTexture(width, height);

  m_texture->cudaRegister();

  initUI();

  m_randomState.seed(time(NULL));
}

void RayMarcher::run()
{
  m_window.show();

  Timer<float> timer;

  while (!m_window.shouldClose())
  {
    // TODO consider refactoring this wait while minimized into the window class
    // itself. Might require running the main loop as a lambda?
    if (m_window.isMinimized())
    {
      m_window.processInput();
      m_window.update();
      continue;
    }

    float deltaTime = timer.elapsed();
    timer.reset();

    process(deltaTime);

    render();

    endAndUpdate();
  }
}

bool RayMarcher::hasCudaSupport()
{
  int n = 0;
  int* count = &n;
  cudaError_t result = cudaGetDeviceCount(count);
  return (result == cudaSuccess && n > 0);
}

void RayMarcher::initBuffers()
{
  std::vector<float> vertices =
  {
    // position             texture coord
    // x     y      z        u     v
     1.0f,  1.0f,  0.0f,    1.0f, 1.0f,
     1.0f, -1.0f,  0.0f,    1.0f, 0.0f,
    -1.0f, -1.0f,  0.0f,    0.0f, 0.0f,
    -1.0f,  1.0f,  0.0f,    0.0f, 1.0f
  };

  std::vector<unsigned int> indices =
  {
      0, 1, 3,
      1, 2, 3
  };

  m_vertexBuffer.data(vertices.data(), vertices.size(), wrndr::BufferUsage::kStaticDraw);
  m_indexBuffer.data(indices.data(), indices.size(), wrndr::BufferUsage::kStaticDraw);

  m_vertexArray.bindElements(m_indexBuffer);

  // positions
  m_vertexArray.bindAttribute<float>(0, m_vertexBuffer, 3, 5 * sizeof(float), 0);

  // texture coordinates
  m_vertexArray.bindAttribute<float>(1, m_vertexBuffer, 2, 5 * sizeof(float), 3 * sizeof(float));
}

void RayMarcher::initTexture(int width, int height)
{
  std::vector<uint8_t> texData(width * height * 4, 255);
  wrndr::TextureCreateInfo createInfo =
  {
    .width = width,
    .height = height,
    .wrapS = wrndr::Wrapping::kClampEdge,
    .wrapT = wrndr::Wrapping::kClampEdge,
    .min = wrndr::Filter::kNearestMipmapNearest,
    .mag = wrndr::Filter::kNearest
  };

  m_texture.emplace(texData, createInfo);
}

void RayMarcher::initUI()
{
  m_paramsInterface = m_interfaceContext.pushLayer<ParamsInterface>();

  m_paramsInterface->setExponentScrollRate(0.3f);

  m_resizeFuncHandle = m_window.registerResizeCallback([this](int width, int height)
    {
      m_resizeDirtyFlag = true;
      m_resizeWidth = width;
      m_resizeHeight = height;
    });
}

void RayMarcher::checkResize()
{
  if (m_paramsInterface->resScaleDirty())
  {
    m_resizeDirtyFlag = true;
    m_resolutionScaling = m_paramsInterface->getResolutionScale();
    m_paramsInterface->resetResScaleDirty();
  }

  if (m_resizeDirtyFlag)
  {
    // TODO properly manage integer division here
    const int newWidth = m_resizeWidth / m_resolutionScaling;
    const int newHeight = m_resizeHeight / m_resolutionScaling;
    m_texture->resize(newWidth, newHeight);
    m_resizeDirtyFlag = false;
  }
}

void RayMarcher::updateMultisamples()
{
  m_numSamples = m_paramsInterface->getSampleCount();
}

void RayMarcher::process(float deltaTime)
{
  m_interfaceContext.tick(deltaTime);

  Camera cam{};
  SphericalCoord coords = m_paramsInterface->getCameraCoords();
  float exponent = m_paramsInterface->getExponent();

  cam.pos = make_float3(coords.x(), coords.y(), coords.z());
  cam.lookAt(make_float3(0.0f, 0.0f, 0.0f));
  float fov = 128.0f / 180.0f * float(3.14159);
  cam.invhalffov = 1.0f / std::tan(fov / 2.0f);

  checkResize();
  updateMultisamples();

  unsigned long long randomSeed = m_distribution(m_randomState);

  switch (m_paramsInterface->getShadingMode())
  {
  case ShadingMode::kDiffuseLight:
  {
    Colour colour = m_paramsInterface->getDiffuseColour();
    m_texture->runKernel(
      compute::rayMarchDiffuseColour,
      cam,
      exponent,
      m_numSamples,
      make_float3(colour.r, colour.g, colour.b),
      randomSeed);
    break;
  }
  case ShadingMode::kNormalColor:
    m_texture->runKernel(
      compute::rayMarchNormalColour,
      cam,
      exponent,
      m_numSamples,
      randomSeed);
    break;
  case ShadingMode::kStepwiseShaded:
  {
    Colour low = m_paramsInterface->getLowStepColour();
    Colour high = m_paramsInterface->getHighStepColour();
    m_texture->runKernel(
      compute::rayMarchStepwiseColour,
      cam,
      exponent,
      m_numSamples,
      make_float3(low.r, low.g, low.b),
      make_float3(high.r, high.g, high.b),
      randomSeed);
    break;
  }
  }

  m_window.processInput();
}

void RayMarcher::render()
{
  m_window.clearColor(0.05f, 0.05f, 0.05f);

  m_interfaceContext.startFrame();

  m_vertexArray.bind();
  m_shaderProgram.bind();
  m_texture->bind();

  wrndr::Renderer::drawIndices<unsigned int>(m_vertexArray, wrndr::Primitive::kTriangles, 6);

  m_interfaceContext.render();
}

void RayMarcher::endAndUpdate()
{
  m_interfaceContext.endFrame();

  m_window.update();
}

} // namespace rmcuda