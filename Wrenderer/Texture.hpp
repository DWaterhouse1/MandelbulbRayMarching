#ifndef WRENDERER_TEXTURE_HPP
#define WRENDERER_TEXTURE_HPP

#include <GLAD/glad.h>
#include <GLFW/glfw3.h>

//std
#include <vector>
#include <cstdint>

namespace wrndr
{

/**
 * Specifies pixel data format.
 */
enum class Format
{
  kDepthComponent = GL_DEPTH_COMPONENT,
  kDepthStencil = GL_DEPTH_STENCIL,
  kRed = GL_RED,
  kRG = GL_RG,
  kRGB = GL_RGB,
  kRGBA = GL_RGBA,
};

/**
 * Specifies internal pixel data config. Must derive from Format.
 */
enum class InternalFormat
{
  kCompressedRed = GL_COMPRESSED_RED,
  kCompressedRedRGTC1 = GL_COMPRESSED_RED_RGTC1,
  kCompressedRG = GL_COMPRESSED_RG,
  kCompressedRGB = GL_COMPRESSED_RGB,
  kCompressedRGBA = GL_COMPRESSED_RGBA,
  kCompressedRGRGTC2 = GL_COMPRESSED_RG_RGTC2,
  kCompressedSignedRedRGTC1 = GL_COMPRESSED_SIGNED_RED_RGTC1,
  kCompressedSignedRGRGTC2 = GL_COMPRESSED_SIGNED_RG_RGTC2,
  kCompressedSRGB = GL_COMPRESSED_SRGB,
  kDepthStencil = GL_DEPTH_STENCIL,
  kDepth24Stencil8 = GL_DEPTH24_STENCIL8,
  kDepth32FStencil8 = GL_DEPTH32F_STENCIL8,
  kDepthComponent = GL_DEPTH_COMPONENT,
  kDepthComponent16 = GL_DEPTH_COMPONENT16,
  kDepthComponent24 = GL_DEPTH_COMPONENT24,
  kDepthComponent32F = GL_DEPTH_COMPONENT32F,
  kR16F = GL_R16F,
  kR16I = GL_R16I,
  kR16SNorm = GL_R16_SNORM,
  kR16UI = GL_R16UI,
  kR32F = GL_R32F,
  kR32I = GL_R32I,
  kR32UI = GL_R32UI,
  kR3G3B2 = GL_R3_G3_B2,
  kR8 = GL_R8,
  kR8I = GL_R8I,
  kR8SNorm = GL_R8_SNORM,
  kR8UI = GL_R8UI,
  kRed = GL_RED,
  kRG = GL_RG,
  kRG16 = GL_RG16,
  kRG16F = GL_RG16F,
  kRG16SNorm = GL_RG16_SNORM,
  kRG32F = GL_RG32F,
  kRG32I = GL_RG32I,
  kRG32UI = GL_RG32UI,
  kRG8 = GL_RG8,
  kRG8I = GL_RG8I,
  kRG8SNorm = GL_RG8_SNORM,
  kRG8UI = GL_RG8UI,
  kRGB = GL_RGB,
  kRGB10 = GL_RGB10,
  kRGB10A2 = GL_RGB10_A2,
  kRGB12 = GL_RGB12,
  kRGB16 = GL_RGB16,
  kRGB16F = GL_RGB16F,
  kRGB16I = GL_RGB16I,
  kRGB16UI = GL_RGB16UI,
  kRGB32F = GL_RGB32F,
  kRGB32I = GL_RGB32I,
  kRGB32UI = GL_RGB32UI,
  kRGB4 = GL_RGB4,
  kRGB5 = GL_RGB5,
  kRGB5A1 = GL_RGB5_A1,
  kRGB8 = GL_RGB8,
  kRGB8I = GL_RGB8I,
  kRGB8UI = GL_RGB8UI,
  kRGB9E5 = GL_RGB9_E5,
  kRGBA = GL_RGBA,
  kRGBA12 = GL_RGBA12,
  kRGBA16 = GL_RGBA16,
  kRGBA16F = GL_RGBA16F,
  kRGBA16I = GL_RGBA16I,
  kRGBA16UI = GL_RGBA16UI,
  kRGBA2 = GL_RGBA2,
  kRGBA32F = GL_RGBA32F,
  kRGBA32I = GL_RGBA32I,
  kRGBA32UI = GL_RGBA32UI,
  kRGBA4 = GL_RGBA4,
  kRGBA8 = GL_RGBA8,
  kRGBA8UI = GL_RGBA8UI,
  kSRGB8 = GL_SRGB8,
  kSRGB8A8 = GL_SRGB8_ALPHA8,
  kSRGBA = GL_SRGB_ALPHA
};

/**
 * Specifies texture wrap mode.
 */
enum class Wrapping
{
  kClampEdge = GL_CLAMP_TO_EDGE,
  kClampBorder = GL_CLAMP_TO_BORDER,
  kRepeat = GL_REPEAT,
  kMirroredRepeat = GL_MIRRORED_REPEAT
};

/**
 * Specifies texture filtering mode.
 */
enum class Filter
{
  kNearest = GL_NEAREST,
  kLinear = GL_LINEAR,
  kNearestMipmapNearest = GL_NEAREST_MIPMAP_NEAREST,
  kLinearMipmapNearest = GL_LINEAR_MIPMAP_NEAREST,
  kNearestMipmapLinear = GL_NEAREST_MIPMAP_LINEAR,
  kLinearMipmapLinear = GL_LINEAR_MIPMAP_LINEAR
};

/**
 * Collection of parameters for construction of a Texture object.
 */
struct TextureCreateInfo
{
  int width = 1;
  int height = 1;
  InternalFormat internalFormat = InternalFormat::kRGBA;
  Format format = Format::kRGBA;
  Wrapping wrapS = Wrapping::kRepeat;
  Wrapping wrapT = Wrapping::kRepeat;
  Filter min = Filter::kLinearMipmapLinear;
  Filter mag = Filter::kLinear;
  bool useMips = true;
};

/**
 * Handle to a texture object. Constructor will copy the data provided it into device memory.
 */
class Texture
{
public:
  Texture(const std::vector<uint8_t>& data, const TextureCreateInfo& createInfo);
  ~Texture();

  Texture(const Texture&) = delete;
  Texture& operator=(const Texture&) = delete;

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
  void setFilters(Filter min, Filter mag);

  /**
   * Resizes the texture object and updates with optional supplied data pointer.
   *
   * @param width New texture width.
   * @param height New texture height.
   * @param data Data to overwrite with.
   */
  void resize(int width, int height, uint8_t* data = nullptr);

  // TODO find a better way to handle this
  [[nodiscard]] GLuint getID() const { return m_id; }
  [[nodiscard]] int getWidth() const { return m_width; }
  [[nodiscard]] int getHeight() const { return m_height; }

private:
  void image2D(const uint8_t* data);

  GLuint m_id = 0;
  int m_width;
  int m_height;
  Format m_format;
  InternalFormat m_internalFormat;
  bool m_useMips;
};
} // namespace wrndr

#endif // WRENDERER_TEXTURE_HPP