#include "UI.hpp"

namespace wrndr
{
UI::UI(Window& window, bool enableDocking)
  : m_usingDocking{ enableDocking }
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  m_io = &ImGui::GetIO();

  if (m_usingDocking)
  {
    m_io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    m_io->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
  }

  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(window.getInternalPtr(), true);
  ImGui_ImplOpenGL3_Init("#version 460");
}

UI::~UI()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

void UI::tick(float deltaTime)
{
  for (auto&& layer : m_interfaceLayers)
  {
    layer->tick(deltaTime);
  }
}

void UI::startFrame()
{
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

void UI::endFrame()
{
  if (m_usingDocking)
  {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
  }
}

void UI::render()
{
  for (auto&& layer : m_interfaceLayers)
  {
    layer->onRender();
  }

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

} // namespace wrndr