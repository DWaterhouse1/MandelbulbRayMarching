#include "ShaderCode.hpp"

namespace wrndr::constants
{
const char* const vertexShaderSource = "#version 460 core\n"
"\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec2 aTexCoord;\n"
"\n"
"out vec2 texCoord;\n"
"\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos, 1.0);\n"
"   texCoord = aTexCoord;\n"
"}";

const char* const fragmentShaderSource = "#version 460 core\n"
"\n"
"out vec4 FragColor;\n"
"\n"
"in vec2 texCoord;\n"
"\n"
"uniform sampler2D inTex;\n"
"\n"
"void main()\n"
"{\n"
"    FragColor = texture(inTex, texCoord);\n"
"}";

} // namespace wrndr::constants