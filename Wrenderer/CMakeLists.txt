cmake_minimum_required (VERSION 3.24)

project(Wrenderer)

set(CMAKE_CXX_STANDARD 20)

find_package(glad CONFIG REQUIRED)

find_package(glm CONFIG REQUIRED)

find_package(glfw3 CONFIG REQUIRED)

find_package(Boost 1.80 REQUIRED)

find_package(ImGui CONFIG REQUIRED)

set(WRENDERER_HEADERS
		Window.hpp
		Buffer.hpp
		VertexArray.hpp
		Shader.hpp
		Program.hpp
		Renderer.hpp
		Texture.hpp
		UI.hpp
		UILayer.hpp
		Types.hpp
		ShaderCode.hpp
		vendor/imgui_impl_glfw.h
		vendor/imgui_impl_opengl3.h
		vendor/imgui_impl_opengl3_loader.h)

set(WRENDERER_SOURCES
		Window.cpp
		VertexArray.cpp
		Shader.cpp
		Program.cpp
		Renderer.cpp
		Texture.cpp
		UI.cpp
		ShaderCode.cpp
		vendor/imgui_impl_glfw.cpp
		vendor/imgui_impl_opengl3.cpp)

add_library(Wrenderer 
	STATIC 
		${WRENDERER_HEADERS}
		${WRENDERER_SOURCES}
		${IMGUI_SOURCES})

target_link_libraries(Wrenderer
	PUBLIC
		imgui::imgui
	PRIVATE
		glad::glad
		glfw
		glm::glm)

target_include_directories(Wrenderer
	PRIVATE
		${GLFW_INCLUDE_DIR}
		${Boost_INCLUDE_DIRS})