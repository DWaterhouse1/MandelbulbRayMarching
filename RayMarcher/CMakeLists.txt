cmake_minimum_required (VERSION 3.24)

project(RayMarcher LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

find_package(Boost 1.80 REQUIRED)

set(RAYMARCHER_HEADERS
		RayMarcher.hpp
		CudaTexture.hpp
		ParamsInterface.hpp
		Timer.hpp
		CudaHelpers.hpp
		ShadingModes.hpp
		RayMarchCompute.cuh
		Shading.cuh)

set(RAYMARCHER_SOURCES
		RayMarcher.cpp
		CudaTexture.cpp
		ParamsInterface.cpp
		RayMarchCompute.cu)

add_library(RayMarcher
	STATIC
		${RAYMARCHER_HEADERS}
		${RAYMARCHER_SOURCES})

target_link_libraries(RayMarcher
		Wrenderer
		CUDA::cudart)

target_include_directories(RayMarcher
	PUBLIC
		${CMAKE_SOURCE_DIR}/Wrenderer
		${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
		${Boost_INCLUDE_DIRS})

set_target_properties(RayMarcher PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_property(TARGET RayMarcher PROPERTY CUDA_ARCHITECTURES native)