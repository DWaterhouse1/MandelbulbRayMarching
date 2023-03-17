#ifndef WRENDERER_PROGRAM_HPP
#define WRENDERER_PROGRAM_HPP

#include <GLAD/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// wrndr
#include "Shader.hpp"

//std
#include <string>

namespace wrndr
{
/**
 * Handle to a shader program object. If Shader objects are provided to constructor, this class will attach and link
 *  them on construction. Otherwise shaders must be attached and linked in user code.
 *
 * @note The program is owned by the driver, so a const instance of this class may manage a non const program on the
 *  device.
 */
class Program
{
public:
  Program();
  explicit Program(const Shader& vertex);
  Program(const Shader& vertex, const Shader& fragment);

  // TODO add proper copy
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  explicit operator GLuint() const { return m_id; }

  ~Program();

  /**
   * Attaches the shader to the program owned by this class.
   *
   * @param shader The shader object to attach.
   */
  void attach(const Shader& shader) const;

  /**
   * Links the attached shaders to produce a bindable program object.
   *
   * @return true on successful link, false otherwise.
   */
  bool link();

  /**
   * Binds the shader program ready for use.
   */
  void bind() const;

  /**
   * Gets the uniform location of the named attribute.
   *
   * @param name Name of the attribute to find.
   * @return Location of the attribute.
   */
  [[nodiscard]] int getAttribute(const std::string& name) const;

  /**
   * Gets the last 512 characters of the link log for this program.
   *
   * @return The link log.
   */
  [[nodiscard]] std::string getLinkLog() const;

  /**
  * Gets the location of the named uniform.
  * 
  * @param name Name of the uniform to find.
  * 
  * @return Location of the uniform.
  */
  [[nodiscard]] int getUniformLocation(const std::string& name) const;

  void setUniform(int location, int value);
  void setUniform(int location, float value);
  void setUniform(int location, bool value);
  void setUniform(int location, const glm::vec2& value);
  void setUniform(int location, const glm::vec3& value);
  void setUniform(int location, const glm::vec4& value);
  void setUniform(int location, const glm::mat2& value, bool transpose = false);
  void setUniform(int location, const glm::mat3& value, bool transpose = false);
  void setUniform(int location, const glm::mat4& value, bool transpose = false);

private:
  GLuint m_id = 0;
  bool m_successfulLink = false;

};
} // namespace wrndr

#endif // WRENDERER_PROGRAM_HPP