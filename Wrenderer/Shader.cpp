#include "Shader.hpp"

namespace wrndr
{
Shader::Shader(ShaderType type)
{
  m_id = glCreateShader(static_cast<GLenum>(type));
}

Shader::Shader(ShaderType type, const std::string& code) : Shader(type)
{
  source(code);
  compile();
}

Shader::~Shader()
{
  glDeleteShader(m_id);
}

void Shader::source(const std::string& code) const
{
  const char* c_str = code.c_str();
  glShaderSource(m_id, 1, &c_str, nullptr);
}

bool Shader::compile()
{
  glCompileShader(m_id);

  GLint result;
  glGetShaderiv(m_id, GL_COMPILE_STATUS, &result);

  m_successfulCompile = (result != GL_FALSE);
  return m_successfulCompile;
}

std::string Shader::getCompileLog() const
{
  GLint result;
  glGetShaderiv(m_id, GL_INFO_LOG_LENGTH, &result);

  if (result > 0)
  {
    std::string log(result, 0);
    glGetShaderInfoLog(m_id, result, &result, log.data());
    return log;
  }
  return "";
}

} // namespace wrndr