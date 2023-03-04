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
  m_resizeHeight{ height }
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

  m_texture->cudaRegister();

  m_paramsInterface = m_interfaceContext.pushLayer<ParamsInterface>();

  m_paramsInterface->setExponentScrollRate(0.3f);

  m_resizeFuncHandle = m_window.registerResizeCallback([this](int width, int height)
    {
      m_resizeDirtyFlag = true;
      m_resizeWidth = width;
      m_resizeHeight = height;
    });
}

void RayMarcher::run()
{
  m_window.show();

  Timer<float> timer;

  while (!m_window.shouldClose())
  {
    float deltaTime = timer.elapsed();
    timer.reset();

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
    
    m_texture->runKernel(
      compute::basicRayMarching,
      cam,
      exponent,
      m_numSamples);

    m_window.processInput();

    m_window.clearColor(0.05f, 0.05f, 0.05f);

    m_interfaceContext.startFrame();

    m_vertexArray.bind();
    m_shaderProgram.bind();
    m_texture->bind();

    wrndr::Renderer::drawIndices<unsigned int>(m_vertexArray, wrndr::Primitive::kTriangles, 6);

    m_interfaceContext.render();
    m_interfaceContext.endFrame();

    m_window.update();
  }
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

} // namespace rmcuda