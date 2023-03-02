#include "CudaTexture.hpp"


namespace rmcuda
{
void CudaInteropTexture::bind()
{
  m_texture.bind();
}

void CudaInteropTexture::setFilters(wrndr::Filter min, wrndr::Filter mag)
{
  m_texture.setFilters(min, mag);
}

void CudaInteropTexture::resize(int width, int height)
{
  cudaUnregister();

  m_texture.resize(width, height);

  cudaRegister();
}

void CudaInteropTexture::cudaRegister()
{
  gpuErrchk(cudaGraphicsGLRegisterImage(
    &m_graphicsResource,
    m_texture.getID(),
    GL_TEXTURE_2D,
    cudaGraphicsRegisterFlagsWriteDiscard));
}

void CudaInteropTexture::cudaUnregister()
{
  gpuErrchk(cudaGraphicsUnregisterResource(m_graphicsResource));
}

void CudaInteropTexture::cudaMap()
{
  gpuErrchk(cudaGraphicsMapResources(1, &m_graphicsResource));
}

void CudaInteropTexture::cudaArray()
{
  gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&m_array, m_graphicsResource, 0, 0));
  m_description.resType = cudaResourceTypeArray;
  m_description.res.array.array = m_array;
}

} // namespace rmcuda