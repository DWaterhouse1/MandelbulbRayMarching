#include "Program.hpp"

//std
#include <iostream>

namespace wrndr
{
Program::Program()
{
  m_id = glCreateProgram();
}

Program::Program(const Shader& vertex) : Program()
{
  attach(vertex);
  link();
  bind();
}

Program::Program(const Shader& vertex, const Shader& fragment) : Program()
{
  attach(vertex);
  attach(fragment);
  link();
  bind();
}

Program::~Program()
{
  glDeleteProgram(m_id);
}

void Program::attach(const Shader& shader) const
{
  glAttachShader(m_id, static_cast<GLuint>(shader));
}

bool Program::link()
{
  glLinkProgram(m_id);

  GLint result;
  glGetProgramiv(m_id, GL_LINK_STATUS, &result);

  if (result != GL_TRUE)
  {
    std::cout << getLinkLog() << "\n";
  }
  else
  {
    m_successfulLink = true;
  }
  return m_successfulLink;
}

void Program::bind() const
{
  //TODO enable only in debug
  if (!m_successfulLink)
  {
    std::cerr << "Error at program: " << m_id << " - can't bind without successful link\n";
  }

  glUseProgram(m_id);
}

int Program::getAttribute(const std::string& name) const
{
  return glGetAttribLocation(m_id, name.c_str());
}

std::string Program::getLinkLog() const
{
  GLint result = 1;
  glGetProgramiv(m_id, GL_INFO_LOG_LENGTH, &result);

  if (result > 0)
  {
    char log[512];
    glGetProgramInfoLog(m_id, 512, nullptr, log);
    std::cout << "Link failed: " << log << "\n";
    return log;
  }
  return "";
}

void Program::setUniform(const int location, const int value)
{
  glUniform1i(location, value);
}

void Program::setUniform(const int location, const float value)
{
  glUniform1f(location, value);
}

void Program::setUniform(const int location, const bool value)
{
  glUniform1i(location, static_cast<int>(value));
}

void Program::setUniform(const int location, const glm::vec2& value)
{
  glUniform2f(location, value.x, value.y);
}

void Program::setUniform(const int location, const glm::vec3& value)
{
  glUniform3f(location, value.x, value.y, value.z);
}

void Program::setUniform(const int location, const glm::vec4& value)
{
  glUniform4f(location, value.x, value.y, value.z, value.w);
}

void Program::setUniform(const int location, const glm::mat2& value, bool transpose)
{
  glUniformMatrix2fv(location, 1, transpose, &value[0][0]);
}

void Program::setUniform(const int location, const glm::mat3& value, bool transpose)
{
  glUniformMatrix3fv(location, 1, transpose, &value[0][0]);
}

void Program::setUniform(const int location, const glm::mat4& value, bool transpose)
{
  glUniformMatrix4fv(location, 1, transpose, &value[0][0]);
}

int Program::getUniformLocation(const std::string& name) const
{
  return glGetUniformLocation(m_id, name.c_str());
}

} // namespace wrndr