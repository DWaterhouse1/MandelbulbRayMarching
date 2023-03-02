#ifndef WRENDERER_UILAYER_HPP
#define WRENDERER_UILAYER_HPP

// imgui
#include "imgui.h"

namespace wrndr
{
class InterfaceLayer
{
public:
	InterfaceLayer() {}
	virtual ~InterfaceLayer() = default;

	InterfaceLayer(const InterfaceLayer&) = delete;
	InterfaceLayer& operator=(const InterfaceLayer&) = delete;

	virtual void onAttach() {}
	virtual void onDettach() {}

	virtual void tick(float deltaTime) {}
	virtual void onRender() {}
};

} // namespace wrndr

#endif // WRENDERER_UILAYER_HPP