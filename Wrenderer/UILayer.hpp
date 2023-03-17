#ifndef WRENDERER_UILAYER_HPP
#define WRENDERER_UILAYER_HPP

// imgui
#include "imgui.h"

namespace wrndr
{
/**
* Interface for a single UI object. Contains UI logic and allows the use of
*		ImGui to display user interface. onRender is pure and must be implemented.
*/
class InterfaceLayer
{
public:
	InterfaceLayer() {}
	virtual ~InterfaceLayer() = default;

	InterfaceLayer(const InterfaceLayer&) = delete;
	InterfaceLayer& operator=(const InterfaceLayer&) = delete;

	/**
	* Called when the layer is added to the stack.
	*/
	virtual void onAttach() {}

	/**
	* TODO layer pop not implemented so this function is unreferenced.
	*/
	virtual void onDettach() {}

	/**
	* Called per frame.
	* 
	* @param deltaTime Elapsed time per frame.
	*/
	virtual void tick(float deltaTime) {}

	/**
	* Executes the UI rendering logic.
	*/
	virtual void onRender() = 0;
};

} // namespace wrndr

#endif // WRENDERER_UILAYER_HPP