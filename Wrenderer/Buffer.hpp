#ifndef WRENDERER_BUFFER_HPP
#define WRENDERER_BUFFER_HPP

#include <GLAD/glad.h>
#include <GLFW/glfw3.h>

//boost
#include <boost/numeric/conversion/cast.hpp>

namespace wrndr
{
/**
 * Describes the usage pattern of buffer data.
 */
enum class BufferUsage
{
  kStreamDraw = GL_STREAM_DRAW,
  kStreamRead = GL_STREAM_READ,
  kStreamCopy = GL_STREAM_COPY,
  kStaticDraw = GL_STATIC_DRAW,
  kStaticRead = GL_STATIC_READ,
  kStaticCopy = GL_STATIC_COPY,
  kDynamicDraw = GL_DYNAMIC_DRAW,
  kDynamicRead = GL_DYNAMIC_READ,
  kDynamicCopy = GL_DYNAMIC_COPY
};

/**
 * Handle and interface to a buffer object.
 *
 * @note Buffer memory is owned by the driver, not this class. A const instance of this class may manage non-const
 *  memory.
 *
 * @tparam T Type contained in the buffer.
 */
template <typename T>
class Buffer
{
public:
  Buffer()
  {
    glGenBuffers(1, &m_id);
  }

  ~Buffer()
  {
    glDeleteBuffers(1, &m_id);
  }

  // not copyable to avoid duplicate delete calls

  //TODO add device data copy for these copy methods
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  operator GLuint() const { return m_id; }

  /**
   * Creates and initializes the array buffer.
   *
   * @param data Pointer to data to copy into the buffer.
   * @param length Length in bytes of the new data.
   * @param usage Usage pattern for the buffer.
   */
  void data(T* data, size_t length, BufferUsage usage) const
  {
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
    glBufferData(
      GL_ARRAY_BUFFER,
      boost::numeric_cast<GLsizeiptr>(length) * sizeof(T),
      static_cast<void*>(data),
      static_cast<GLenum>(usage));
  }

  /**
   * Updates a subset of the array buffer.
   *
   * @param data Pointer to data to copy into the buffer.
   * @param offset Offset in bytes to where the new data will be copied in.
   * @param length Length in bytes of the new data copy.
   */
  void subData(T* data, size_t offset, size_t length)
  {
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
    glBufferSubData(
      GL_ARRAY_BUFFER,
      boost::numeric_cast<GLsizeiptr>(offset),
      boost::numeric_cast<GLsizeiptr>(length),
      static_cast<void*>(data));
  }

  /**
   * Returns a subset of the array buffer.
   *
   * @param[out] data Pointer to the returned data.
   * @param[in] offset Offset in bytes to the beginning of returned data.
   * @param[in] length Length in bytes of the returned data region.
   */
  void getSubData(T* data, size_t offset, size_t length)
  {
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
    glGetBufferSubData(
      GL_ARRAY_BUFFER,
      boost::numeric_cast<GLsizeiptr>(offset),
      boost::numeric_cast<GLsizeiptr>(length),
      static_cast<void*>(data));
  }

  /**
   * Evaluates if this Buffer is currently associated with a valid device buffer.
   *
   * @return True if there is a valid associated device buffer.
   */
  [[nodiscard]] bool isValid() const { return glIsBuffer(m_id); }

private:
  GLuint m_id = 0;
};
} // namespace wrndr

#endif // WRENDERER_BUFFER_HPP