#ifndef WRENDERER_UI_HPP
#define WRENDERER_UI_HPP

// imgui
#include "imgui.h"
#include "vendor/imgui_impl_glfw.h"
#include "vendor/imgui_impl_opengl3.h"

// wrndr
#include "Window.hpp"
#include "UILayer.hpp"

//std
#include <vector>
#include <memory>
#include <list>
#include <type_traits>

namespace wrndr
{
class UI
{
public:
	UI(Window& window, bool enableDocking = true);
	~UI();

	UI(const UI&) = delete;
	UI& operator=(const UI&) = delete;

	void tick(float deltaTime);
	void startFrame();
	void endFrame();
	void render();

	template<typename T, typename... Args>
	T* pushLayer(Args... args)
	{
		static_assert(
			std::is_base_of<InterfaceLayer, T>::value,
			"UI.hpp, UI::pushLayer Error : supplied type is not subclass of InterfaceLayer!");

		std::unique_ptr<T> newLayer = std::make_unique<T>(std::forward<Args>(args)...);

		T* retLayer = newLayer.get();

		newLayer->onAttach();

		m_interfaceLayers.push_back(std::move(newLayer));

		return retLayer;
	}

private:
	ImGuiIO* m_io = nullptr;
	bool m_usingDocking;

	std::list<std::unique_ptr<InterfaceLayer>> m_interfaceLayers;
};

} // namespace rm

#endif // WRENDERER_UI_HPP