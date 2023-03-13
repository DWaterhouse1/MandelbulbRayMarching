#ifndef WRENDERER_WINDOW_HPP
#define WRENDERER_WINDOW_HPP

#include <GLAD/glad.h>
#include <GLFW/glfw3.h>

//std
#include <string>
#include <functional>
#include <map>

namespace wrndr
{
/**
 * Implements a window to which images can be presented and manages the context. The window can clear an RGB colour
 *  per frame, and can handle received inputs.
 */
class Window
{
public:
  explicit Window(std::string title = "glfw window", int width = 800, int height = 600);

  ~Window();

  // not copyable
  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;

  /**
   * Sets the color buffer clear value.
   *
   * @param r Red component.
   * @param g Green component.
   * @param b Blue component.
   */
  void clearColor(float r, float g, float b);

  /**
   * Processes input received by the window. Window will close on ESC input.
   */
  void processInput();

  /**
   * Swaps the present buffers and polls window input.
   */
  void update();

  /**
   * Finds if the window wants to close.
   *
   * @return True if the window wants to close.
   */
  [[nodiscard]] bool shouldClose() const;

  /**
   * Makes the window visible.
   */
  void show();

  //TODO clean this up
  [[nodiscard]] GLFWwindow* getInternalPtr() const { return m_window; }

  /**
  * Checks if this window object is minimized.
  * 
  * @return If the window is minimzed.
  */
  [[nodiscard]] bool isMinimized() const;

  /**
  *
  */
  unsigned int registerResizeCallback(std::function<void(int, int)> callback);

  /**
  *
  */
  void deregisterResizeCallback(unsigned int& handle);

private:
  void resizeCallback(int width, int height);
  static void resizeCallbackImpl(GLFWwindow* window, int width, int height);

  GLFWwindow* m_window = nullptr;
  const std::string m_title;
  int m_width;
  int m_height;

  std::map<unsigned int, std::function<void(int, int)>> m_resizeCallbackList;

};
} // namespace wrndr

#endif // WRENDERER_WINDOW_HPP