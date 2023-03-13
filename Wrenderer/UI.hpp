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
			"Pushed type is not subclass of InterfaceLayer!");

		T* newLayer = static_cast<T*>(m_interfaceLayers.emplace_back(std::make_unique<T>(std::forward<Args>(args)...)).get());

		//m_interfaceLayers.push_back(newLayer);

		newLayer->onAttach();

		return newLayer;
	}

private:
	ImGuiIO* m_io = nullptr;
	bool m_usingDocking;

	std::list<std::unique_ptr<InterfaceLayer>> m_interfaceLayers;
};

} // namespace rm

#endif // WRENDERER_UI_HPP