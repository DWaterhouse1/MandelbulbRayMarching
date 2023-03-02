#ifndef WRENDERER_SHADER_HPP
#define WRENDERER_SHADER_HPP

#include <GLAD/glad.h>
#include <GLFW/glfw3.h>

// std
#include <string>

namespace wrndr
{
/**
 * Describes the shader type.
 */
enum class ShaderType
{
  kCompute = GL_COMPUTE_SHADER,
  kVertex = GL_VERTEX_SHADER,
  kTesselationControl = GL_TESS_CONTROL_SHADER,
  kTesselationEvaluation = GL_TESS_EVALUATION_SHADER,
  kGeometry = GL_GEOMETRY_SHADER,
  kFragment = GL_FRAGMENT_SHADER
};

/**
 * Handle to a shader object, with creation and compilation interface.
 *
 * @note Shader program is owned by the driver, so a const instance of this class may refer to a non const shader
 *  on the device.
 */
class Shader
{
public:
  explicit Shader(ShaderType type);
  Shader(ShaderType type, const std::string& code);

  //TODO add proper copy
  Shader(const Shader&) = delete;
  Shader& operator=(const Shader&) = delete;

  ~Shader();

  explicit operator GLuint() const { return m_id; }

  /**
   * Uploads the source code used to define this shader program.
   *
   * @param code The shader source code.
   */
  void source(const std::string& code) const;

  /**
   * Compiles the shader.
   *
   * @return True on successful compilation.
   */
  bool compile();

  /**
   * Gets the info log associated with the shader compilation.
   *
   * @return Shader info log. Empty string returned on successful compilation.
   */
  [[nodiscard]] std::string getCompileLog() const;

  /**
   * Evaluates if this shader has been compiled successfully.
   *
   * @return True if shader has been compiled and is ready to link in a program.
   */
  [[nodiscard]] bool isValid() const { return m_successfulCompile; }

private:
  GLuint m_id = 0;
  bool m_successfulCompile = false;
};
} // namespace wrndr

#endif // WRENDERER_SHADER_HPP