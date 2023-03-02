#include "Renderer.hpp"

namespace wrndr
{
void Renderer::draw(const VertexArray& vao, Primitive primitive, int offset, int vertices)
{
  glBindVertexArray(static_cast<GLuint>(vao));
  glDrawArrays(static_cast<GLenum>(primitive), offset, vertices);
}

} // namespace wrndr