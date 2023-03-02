#ifndef WRENDERER_TYPES_HPP
#define WRENDERER_TYPES_HPP

#include <GLAD/glad.h>
#include <GLFW/glfw3.h>

namespace wrndr
{
/**
 * Provides implementation recognised type information from standard fundamental types.
 *
 * @tparam T Fundamental type to use.
 */
template<typename T>
struct FundamentalType
{
  static constexpr int value;
};

template<>
struct FundamentalType<char>
{
  static constexpr int value = GL_BYTE;
};

template<>
struct FundamentalType<unsigned char>
{
  static constexpr int value = GL_UNSIGNED_BYTE;
};

template<>
struct FundamentalType<int16_t>
{
  static constexpr int value = GL_SHORT;
};

template<>
struct FundamentalType<uint16_t>
{
  static constexpr int value = GL_UNSIGNED_SHORT;
};

template<>
struct FundamentalType<int32_t>
{
  static constexpr int value = GL_INT;
};

template<>
struct FundamentalType<uint32_t>
{
  static constexpr int value = GL_UNSIGNED_INT;
};

template<>
struct FundamentalType<float>
{
  static constexpr int value = GL_FLOAT;
};

template<>
struct FundamentalType<double>
{
  static constexpr int value = GL_DOUBLE;
};

} // namespace wrndr

#endif // WRENDERER_TYPES_HPP