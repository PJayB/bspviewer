static const char * vertSrc = R"GLSL(
#version 120
uniform mat4 matrix;
attribute vec4 vertex;
attribute vec3 normal;
attribute vec4 texcoord;
attribute vec4 lmcoord1;
attribute vec4 lmcoord2;
attribute vec4 lmcoord3;
attribute vec4 lmcoord4;
varying vec3 fragNormal;

// todo: vertex colors

void main()
{
	gl_TexCoord[0] = texcoord;
	gl_TexCoord[1] = lmcoord1;
	gl_TexCoord[2] = lmcoord2;
	gl_TexCoord[3] = lmcoord3;
	gl_TexCoord[4] = lmcoord4;
	gl_Position = matrix * vertex;
	fragNormal = normal;
}
)GLSL";

static const char * fragSrc = R"GLSL(
#version 120
varying vec3 fragNormal;
uniform sampler2D texture;
uniform sampler2D lightmap1;
uniform sampler2D lightmap2;
uniform sampler2D lightmap3;
uniform sampler2D lightmap4;
uniform float lightmapGamma;

// todo: really should be separate shaders for perf
void main() {
	vec4 texel = texture2D(texture, gl_TexCoord[0].st);
	vec4 lm = texture2D(lightmap1, gl_TexCoord[1].st); // todo fixme
			/* + texture2D(lightmap2, gl_TexCoord[2].st)
			+ texture2D(lightmap3, gl_TexCoord[3].st)
			+ texture2D(lightmap4, gl_TexCoord[4].st); */
	texel = texel * lm * lightmapGamma;
	gl_FragColor = texel;
}
)GLSL";
