#ifndef WRENDERER_VERTEXARRAY_HPP
#define WRENDERER_VERTEXARRAY_HPP

#include <GLAD/glad.h>
#include <GLFW/glfw3.h>

// wrndr
#include "Buffer.hpp"
#include "Types.hpp"

// boost
#include <boost/numeric/conversion/cast.hpp>

namespace wrndr
{
/**
 * Handle to a vertex array object.
 *
 * @note Vertex array is owned by the driver, so a const instance of this class may refer to a non const vao object
 *  on the device.
 */
class VertexArray
{
public:
  VertexArray();

  VertexArray(const VertexArray&) = delete;
  VertexArray& operator=(const VertexArray&) = delete;

  ~VertexArray();

  explicit operator GLuint() const { return m_id; }

  /**
   * Binds the Vertex pointer of this Array object.
   *
   * @tparam T Type contained in the buffer.
   * @param attribute Location of the attribute to bind.
   * @param buffer The array buffer.
   * @param count Number of components per vertex attribute.
   * @param stride Number of bytes between consecutive vertex attributes.
   * @param offset Start in bytes of the vertex attribute.
   */
  template<typename T>
  void bindAttribute(const int attribute, const Buffer<T>& buffer, size_t count, size_t stride, size_t offset)
  {
    // regrettable, but required due to glVertexAttribPointer asking for a void* type for the offset
    const void* offsetPointer = static_cast<char const*>(0) + offset;

    glBindVertexArray(m_id);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glEnableVertexAttribArray(attribute);
    glVertexAttribPointer(
      attribute,
      boost::numeric_cast<GLint>(count),
      FundamentalType<T>::value,
      GL_FALSE,
      boost::numeric_cast<GLint>(stride),
      offsetPointer);
  }

  /**
   * Binds the supplied element buffer.
   *
   * @tparam T Type contained by the element buffer.
   * @param elements Buffer of elements.
   */
  template<typename T>
  void bindElements(const Buffer<T>& elements)
  {
    glBindVertexArray(m_id);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elements);
  }

  /**
   * Binds the owned array.
   */
  void bind() const;

private:
  GLuint m_id = 0;
};
} // namespace wrndr

#endif // WRENDERER_VERTEXARRAY_HPP