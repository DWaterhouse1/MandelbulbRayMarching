#include "Texture.hpp"

namespace wrndr
{
namespace
{
/**
 * RAII object which saves texture binding state on construct, and restores state on destruct.
 */
class TextureBindingSaver
{
public:
  TextureBindingSaver()
  {
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &m_id);
  }

  ~TextureBindingSaver()
  {
    glBindTexture(GL_TEXTURE_2D, m_id);
  }

  TextureBindingSaver(const TextureBindingSaver&) = delete;
  TextureBindingSaver& operator=(const TextureBindingSaver&) = delete;

  TextureBindingSaver(const TextureBindingSaver&&) = delete;
  TextureBindingSaver& operator=(const TextureBindingSaver&&) = delete;

private:
  GLint m_id = 0;
};
} // namespace

Texture::~Texture()
{
  glDeleteTextures(1, &m_id);
}

Texture::Texture(const std::vector<uint8_t>& data, const TextureCreateInfo& createInfo) :
  m_height{ createInfo.height },
  m_width{ createInfo.width },
  m_format{ createInfo.format },
  m_internalFormat{ createInfo.internalFormat },
  m_useMips{ createInfo.useMips }
{
  TextureBindingSaver saved;

  glGenTextures(1, &m_id);
  glBindTexture(GL_TEXTURE_2D, m_id);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, static_cast<GLint>(createInfo.wrapS));
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, static_cast<GLint>(createInfo.wrapT));
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(createInfo.min));
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(createInfo.mag));

  image2D(data.data());

  if (m_useMips) glGenerateMipmap(GL_TEXTURE_2D);
}

void Texture::bind()
{
  glBindTexture(GL_TEXTURE_2D, m_id);
}

void Texture::setFilters(Filter min, Filter mag)
{
  TextureBindingSaver saved;

  glBindTexture(GL_TEXTURE_2D, m_id);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(min));
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(mag));
}

void Texture::resize(int width, int height, uint8_t* data)
{
  TextureBindingSaver saved;

  glBindTexture(GL_TEXTURE_2D, m_id);

  m_width = width;
  m_height = height;

  image2D(data);

  if (m_useMips) glGenerateMipmap(GL_TEXTURE_2D);
}

void Texture::image2D(const uint8_t* data)
{
  glTexImage2D(
    GL_TEXTURE_2D,
    0,
    static_cast<GLint>(m_internalFormat),
    m_width,
    m_height,
    0,
    static_cast<GLenum>(m_format),
    GL_UNSIGNED_BYTE,
    data);
}

} // namespace wrndr