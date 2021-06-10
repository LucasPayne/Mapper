#version 420

in TES_OUT {
    vec2 uv;
} f_in;

uniform sampler2D rgb_image;

out vec4 color;

void main(void)
{
    color = texture(rgb_image, f_in.uv);
}
