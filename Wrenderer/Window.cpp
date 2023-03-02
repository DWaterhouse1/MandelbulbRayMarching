#include "Window.hpp"

//std
#include <iostream>

namespace wrndr
{
namespace
{
void debugMessageCallback(
  GLenum source,
  GLenum type,
  GLuint id,
  GLenum severity,
  GLsizei length,
  const GLchar* message,
  const void* userParam)
{
  if (severity != GL_DEBUG_SEVERITY_NOTIFICATION)
  {
    std::string debugMessage(message);
    std::cout << debugMessage << "\n";
  }
}
} // namespace

Window::Window(std::string title, int width, int height) :
  m_title{ std::move(title) },
  m_width{ width },
  m_height{ height }
{
  if (!glfwInit())
  {
    std::cerr << "Error: failed to initialize glfw\n";
    // TODO init failed, consider refactor to builder that can return optional
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  // start hidden, show() to show
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

  m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
  if (m_window == nullptr)
  {
    std::cerr << "Failed to create glfw window\n";
    glfwTerminate();
    // TODO init failed, consider refactor to builder that can return optional
  }

  glfwMakeContextCurrent(m_window);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cerr << "Failed to initialize GLAD\n";
    // TODO init failed, consider refactor to builder that can return optional
  }

  glfwSetWindowUserPointer(m_window, reinterpret_cast<void*>(this));

  glViewport(0, 0, m_width, m_height);

  glfwSetFramebufferSizeCallback(m_window, resizeCallbackImpl);

#ifdef _DEBUG
  glEnable(GL_DEBUG_OUTPUT);
#endif
  glDebugMessageCallback(debugMessageCallback, nullptr);

  // good initialize if we make it here
}

Window::~Window()
{
  // TODO destroy only on good init
  glfwTerminate();

  if (m_window != nullptr) glfwDestroyWindow(m_window);
}

void Window::clearColor(float r, float g, float b)
{
  glClearColor(r, g, b, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
}

void Window::processInput()
{
  if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
  {
    glfwSetWindowShouldClose(m_window, GL_TRUE);
  }
}

void Window::update()
{
  glfwSwapBuffers(m_window);
  glfwPollEvents();
}

bool Window::shouldClose() const
{
  return glfwWindowShouldClose(m_window);
}

void Window::show()
{
  glfwShowWindow(m_window);
}

void Window::resizeCallback(int width, int height)
{
  glViewport(0, 0, width, height);

  for (const auto& [handle, func] : m_resizeCallbackList)
  {
    func(width, height);
  }
}

void Window::resizeCallbackImpl(GLFWwindow* window, int width, int height)
{
  Window* wrapper = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  wrapper->resizeCallback(width, height);
}

unsigned int Window::registerResizeCallback(std::function<void(int, int)> callback)
{
  static unsigned int handle = 1;
  m_resizeCallbackList.insert({ handle, callback });
  return handle++;
}

void Window::deregisterResizeCallback(unsigned int& handle)
{
  if (handle > 0 && m_resizeCallbackList.contains(handle))
  {
    m_resizeCallbackList.erase(handle);
  }

  handle = 0;
}

} // namespace wrndr