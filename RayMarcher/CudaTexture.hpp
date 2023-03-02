#ifndef RAYMARCHER_CUDATEXTURE_HPP
#define RAYMARCHER_CUDATEXTURE_HPP

// wrenderer
#include "Texture.hpp"

// cuda
#include <cuda_gl_interop.h>

// helpers
#include "CudaHelpers.hpp"

namespace rmcuda
{
/**
 * Wrapper for Texture which allows for registration of memory with CUDA context.
 */
class CudaInteropTexture
{
public:
  CudaInteropTexture(const std::vector<uint8_t>& data, const wrndr::TextureCreateInfo& createInfo)
    : m_texture(data, createInfo) {}

  // ------------- forwarding -------------
  /**
   * Binds this texture for use in draw calls.
   */
  void bind();

  /**
   * Sets the filtering modes for this texture.
   *
   * @param min Filter to use for texture minifying.
   * @param mag Filter to use for texture magnifying.
   */
  void setFilters(wrndr::Filter min, wrndr::Filter mag);

  /**
   * Resizes the texture object and updates with optional supplied data pointer.
   *
   * @param width New texture width.
   * @param height New texture height.
   * @param data Data to overwrite with.
   */
  void resize(int width, int height);
  // --------------------------------------

  /**
   * Registes this texture memory with CUDA. OpenGL calls using this texture
   *   should not be made while it is registered in this way.
   */
  void cudaRegister();

  /**
   * Unregisters the texture memory so it is not visible to CUDA. Kernals should
   *   not be run on this texture while unregistered.
   */
  void cudaUnregister();

  /**
   * Runs the provided kernel wrapper with this textures write surface. The
   * first and second arguments of the kernel must be of type cudaSurfaceObject_t
   * and dim3 respectively, so that the kernal may run appropriately using this
   * texture as a write surface.
   *
   * @param kernel Kernel function pointer.
   * @param args kernel arguments.
   *
   * @tparam Fn Kernel function pointer type.
   * @tparam Args Kernel arguments.
   */
  template<typename Fn, typename... Args>
  void runKernel(Fn kernel, Args... args)
  {
    cudaMap();
    cudaArray();

    cudaSurfaceObject_t writeSurface;
    gpuErrchk(cudaCreateSurfaceObject(&writeSurface, &m_description));

    dim3 dim(m_texture.getWidth(), m_texture.getHeight());

    kernel(writeSurface, dim, std::forward<Args>(args)...);

    gpuErrchk(cudaDestroySurfaceObject(writeSurface));
    gpuErrchk(cudaGraphicsUnmapResources(1, &m_graphicsResource));
    gpuErrchk(cudaStreamSynchronize(0));
  }

private:
  void cudaMap();
  void cudaArray();

  wrndr::Texture m_texture;
  cudaGraphicsResource_t m_graphicsResource = nullptr;
  cudaArray_t m_array = nullptr;
  cudaResourceDesc m_description{};
};
} // namespace rmcuda

#endif // RAYMARCHER_CUDATEXTURE_HPP