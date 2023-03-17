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
/**
* Defines a UI context, which manages and displays a stack of objects derived
*		from InterfaceLayer.
*/
class UI
{
public:
	UI(Window& window, bool enableDocking = true);
	~UI();

	UI(const UI&) = delete;
	UI& operator=(const UI&) = delete;

	/**
	* Calls the tick function on all held InterfaceLayer objects.
	* 
	* @deltaTime Elapsed time.
	*/
	void tick(float deltaTime);

	/**
	* Brings up the ImGui context for rendering a frame.
	*/
	void startFrame();

	/**
	* Finishes ImGui rendering for a frame.
	*/
	void endFrame();

	/**
	* Executes ImGui render logic.
	*/
	void render();

	/**
	* Creates and adds an InterfaceLayer derived object. The object is owned by
	*		this class and kept alive until it is popped or this class is destroyed.
	*		Additionally calls the layer's onAttach method.
	* 
	* @param args The argument list for the constructor of the new layer.
	* 
	* @tparam T the type of the new layer. Must derive publicly from InterfaceLayer
	* @tparam Args Parameter pack for the constructor of T.
	* 
	* @return Pointer to the newly constructed layer.
	*/
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