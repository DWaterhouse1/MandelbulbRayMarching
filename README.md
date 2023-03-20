# Mandelbulb Ray Marching

A simple toy project to visualise the mandelbulb fractal with standard ray marching methods. Runs on CUDA with OpenGL for real time presentation.

### Dependencies

- NVIDIA CUDA Toolkit
- OpenGL
- GLM
- GLAD
- glfw3
- ImGui
- Boost: Optional and Numeric Conversions

Built with CMake, but does not export a target. You may need to manually manage the dependencies if you would like to build locally.

### RayMarcher

Contains the logic to run the ray marching algorithm. Uses cuda to draw on a registered GL texture.

### Wrenderer

Very thin OpenGL abstraction.

### Performance concerns

The ray marching algorithm runs at interactive rates, but is very intensive, at least as implemented. May be prohibitively slow on older GPUs. The performance is highly dependant on the multisampling and resolution scaling options. The stepwise shading mode runs significantly faster, since it doesn't need to calculate normalised gradient values of the SDF like the diffuse and normal shading modes.
