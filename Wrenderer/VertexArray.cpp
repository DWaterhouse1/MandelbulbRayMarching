#include "VertexArray.hpp"

namespace wrndr
{
VertexArray::VertexArray()
{
  glGenVertexArrays(1, &m_id);
}

VertexArray::~VertexArray()
{
  glDeleteVertexArrays(1, &m_id);
}

void VertexArray::bind() const
{
  glBindVertexArray(m_id);
}

} // namespace wrndr