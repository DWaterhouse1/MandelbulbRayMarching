#ifndef WRENDERER_RENDERER_HPP
#define WRENDERER_RENDERER_HPP

#include <GLAD/glad.h>
#include <GLFW/glfw3.h>

// wrndr
#include "VertexArray.hpp"
#include "Types.hpp"

//std
#include <concepts>

namespace wrndr
{
/**
 * Describes the primitive used to draw vertex data.
 */
enum class Primitive
{
  kPoints = GL_POINTS,
  kLineStrip = GL_LINE_STRIP,
  kLineLoop = GL_LINE_LOOP,
  kLines = GL_LINES,
  kLineStripAdjacency = GL_LINE_STRIP_ADJACENCY,
  kLinesAdjacency = GL_LINES_ADJACENCY,
  kTriangleStrip = GL_TRIANGLE_STRIP,
  kTriangleFan = GL_TRIANGLE_FAN,
  kTriangles = GL_TRIANGLES,
  kTriangleStripAdjacency = GL_TRIANGLE_STRIP_ADJACENCY,
  kTrianglesAdjacency = GL_TRIANGLES_ADJACENCY
};

namespace Renderer
{
namespace details
{
template <typename T>
concept gl_index_type =
std::same_as<T, unsigned char> or
std::same_as<T, unsigned int> or
std::same_as<T, unsigned short int>;
} // namespace

/**
 * Makes a draw call on the currently bound vertex data.
 *
 * @param vao Object specifying the vertex data buffers to be used for drawing.
 * @param primitive Primitive draw mode.
 * @param offset Offset into the vertex arrays to start from.
 * @param vertices Vertex count to draw.
 */
void draw(const VertexArray& vao, Primitive primitive, int offset, int vertices);

/**
 * Makes an indexed draw call on the currently bound vertex and index data.
 *
 * @tparam T Type of the index data used.
 * @param vao Object specifying the vertex/index data buffers to be used for drawing.
 * @param primitive Primitive draw mode.
 * @param count Index count to draw.
 * @param indices Pointer to indices
 *
 * @note Index data type must be uchar, uint, or ushort.
 */
template <details::gl_index_type T>
void drawIndices(const VertexArray& vao, Primitive primitive, int count, int* indices = nullptr)
{
  glBindVertexArray(static_cast<GLuint>(vao));
  glDrawElements(static_cast<GLenum>(primitive), count, FundamentalType<T>::value, indices);
}

} // namespace Renderer
} // namespace wrndr

#endif // WRENDERER_RENDERER_HPP