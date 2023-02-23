#include <cmath>
#include <cstddef>
#include <array>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <physfs.h>
#include <memory.h>
#include "filestream.hpp"
#include "bsp.hpp"

enum
{
    ENTITY = 0,
    SHADER,
    PLANE,
    NODE,
    LEAF,
    LEAFFACE,
    LEAFBRUSH,
    MODEL,
    BRUSH,
    BRUSHSIDE,
    VERTEX,
    MESHVERTEX,
    EFFECT,
    FACE,
    LIGHTMAP,
    LIGHTVOL,
    VISDATA,
    NUM_LUMPS_IBSP,

    LIGHTARRAY = NUM_LUMPS_IBSP, // RBSP only
    NUM_LUMPS_RBSP,

    MAX_LUMPS = NUM_LUMPS_RBSP
};

const int CONTENTS_SOLID        = 0x1;
const int CONTENTS_LAVA         = 0x8;
const int CONTENTS_SLIME        = 0x10;
const int CONTENTS_WATER        = 0x20;
const int CONTENTS_FOG          = 0x40;
const int CONTENTS_NOTTEAM1     = 0x80;
const int CONTENTS_NOTTEAM2     = 0x100;
const int CONTENTS_NOBOTCLIP    = 0x200;
const int CONTENTS_AREAPORTAL   = 0x8000;
const int CONTENTS_PLAYERCLIP   = 0x10000;
const int CONTENTS_MONSTERCLIP  = 0x20000;
const int CONTENTS_TELEPORTER   = 0x40000;
const int CONTENTS_JUMPPAD      = 0x80000;
const int CONTENTS_CLUSTERPORTAL= 0x100000;
const int CONTENTS_DONOTENTER   = 0x200000;
const int CONTENTS_BOTCLIP      = 0x400000;
const int CONTENTS_MOVER        = 0x800000;
const int CONTENTS_ORIGIN       = 0x1000000;
const int CONTENTS_BODY         = 0x2000000;
const int CONTENTS_CORPSE       = 0x4000000;
const int CONTENTS_DETAIL       = 0x8000000;
const int CONTENTS_STRUCTURAL   = 0x10000000;
const int CONTENTS_TRANSLUCENT  = 0x20000000;
const int CONTENTS_TRIGGER      = 0x40000000;
const int CONTENTS_NODROP       = 0x80000000;

const int SURF_NODAMAGE     = 0x1;
const int SURF_SLICK        = 0x2;
const int SURF_SKY          = 0x4;
const int SURF_LADDER       = 0x8;
const int SURF_NOIMPACT     = 0x10;
const int SURF_NOMARKS      = 0x20;
const int SURF_FLESH        = 0x40;
const int SURF_NODRAW       = 0x80;
const int SURF_HINT         = 0x100;
const int SURF_SKIP         = 0x200;
const int SURF_NOLIGHTMAP   = 0x400;
const int SURF_POINTLIGHT   = 0x800;
const int SURF_METALSTEPS   = 0x1000;
const int SURF_NOSTEPS      = 0x2000;
const int SURF_NONSOLID     = 0x4000;
const int SURF_LIGHTFILTER  = 0x8000;
const int SURF_ALPHASHADOW  = 0x10000;
const int SURF_NODLIGHT     = 0x20000;
const int SURF_SURFDUST     = 0x40000;

#define LIGHTMAP_2D			-4		// shader is for 2D rendering
#define LIGHTMAP_BY_VERTEX	-3		// pre-lit triangle models
#define LIGHTMAP_WHITEIMAGE	-2
#define	LIGHTMAP_NONE		-1

const void* VertexPosition = (void*)(long)offsetof(Vertex, position);
const void* VertexTexCoord = (void*)(long)offsetof(Vertex, texCoord);
const void* VertexLMCoord0 = (void*)(long)offsetof(Vertex, lmCoord[0]);
const void* VertexLMCoord1 = (void*)(long)offsetof(Vertex, lmCoord[1]);
const void* VertexLMCoord2 = (void*)(long)offsetof(Vertex, lmCoord[2]);
const void* VertexLMCoord3 = (void*)(long)offsetof(Vertex, lmCoord[3]);
const void* VertexNormalCoord = (void*)(long)offsetof(Vertex, normal);

struct Lump
{
    int offset;
    int size;
};

struct Header
{
    char magic[4];
    int version;
};

struct RawShader
{
    char name[64];
    int surface;
    int contents;
};

struct RawBrushSide_IBSP
{
    int plane;
    int shader;
};

struct RawVertex_IBSP {
    glm::vec3 position;
    glm::vec2 texCoord;
    glm::vec2 lmCoord;
    glm::vec3 normal;
    unsigned char colour[4];
};

struct RawFace_IBSP
{
    int shader;
    int effect;
    int type;
    int vertexOffset;
    int vertexCount;
    int meshVertexOffset;
    int meshVertexCount;
    int lightMap;
    int lightMapStart[2];
    int lightMapSize[2];
    glm::vec3 lightMapOrigin;
    glm::vec3 lightMapVecs[2];
    glm::vec3 normal;
    int size[2];
};

struct RawFace_RBSP
{
    int shader;
    int effect;
    int type;
    int vertexOffset;
    int vertexCount;
    int meshVertexOffset;
    int meshVertexCount;
    unsigned char lightMapStyles[MaxLightMapCount];
    unsigned char vertexStyles[MaxLightMapCount];
    int lightMaps[MaxLightMapCount];
    int lightMapStart[2][MaxLightMapCount];
    int lightMapSize[2];
    glm::vec3 lightMapOrigin;
    glm::vec3 lightMapVecs[2];
    glm::vec3 normal;
    int size[2];
};

struct RawLightVol_IBSP
{
    unsigned char ambient[3];
    unsigned char directional[3];
    unsigned char direction[2];
};

struct RawLightVol_RBSP
{
    unsigned char ambient[MaxLightMapCount][3];
    unsigned char directional[MaxLightMapCount][3];
    unsigned char styles[MaxLightMapCount];
    unsigned char direction[2];
};

template<typename T> void copy_array(T& dst, const T& src)
{
    memcpy(dst, src, sizeof(src));
}

static BrushSide ibsp_to_rbsp(const RawBrushSide_IBSP& b)
{
    BrushSide a;
    memset(&a, 0, sizeof(a));

    a.plane = b.plane;
    a.shader = b.shader;
    a.drawSurf = -1;
    return a;
}

static Vertex ibsp_to_rbsp(const RawVertex_IBSP& b)
{
    Vertex a;
    memset(&a, 0, sizeof(a));

    a.position = b.position;
    a.texCoord = b.texCoord;
    a.lmCoord[0] = b.lmCoord;
    a.normal = b.normal;
    return a;
}

static RawFace_RBSP ibsp_to_rbsp(const RawFace_IBSP& b)
{
    RawFace_RBSP a;
    memset(&a, 0, sizeof(a));

    a.shader = b.shader;
    a.effect = b.effect;
    a.type = b.type;
    a.vertexOffset = b.vertexOffset;
    a.vertexCount = b.vertexCount;
    a.meshVertexOffset = b.meshVertexOffset;
    a.meshVertexCount = b.meshVertexCount;
    a.lightMaps[0] = b.lightMap;
    for (int i = 1; i < MaxLightMapCount; ++i)
        a.lightMaps[i] = LIGHTMAP_NONE;
    a.lightMapStart[0][0] = b.lightMapStart[0];
    a.lightMapStart[1][0] = b.lightMapStart[1];
    copy_array(a.lightMapSize, b.lightMapSize);
    a.lightMapOrigin = b.lightMapOrigin;
    copy_array(a.lightMapVecs, b.lightMapVecs);
    a.normal = b.normal;
    copy_array(a.size, b.size);
    return a;
}

static RawLightVol_RBSP ibsp_to_rbsp(const RawLightVol_IBSP& b)
{
    RawLightVol_RBSP a;
    memset(&a, 0, sizeof(a));

    copy_array(a.ambient[0], b.ambient);
    copy_array(a.directional[0], b.directional);
    copy_array(a.direction, b.direction);
    return a;
}

Vertex operator+(const Vertex& v1, const Vertex& v2)
{
    Vertex temp;
    temp.position = v1.position + v2.position;
    temp.texCoord = v1.texCoord + v2.texCoord;
    for (int i = 0; i < MaxLightMapCount; ++i) {
        temp.lmCoord[i] = v1.lmCoord[i] + v2.lmCoord[i];
    }
    temp.normal = v1.normal + v2.normal;
    return temp;
}

Vertex operator*(const Vertex& v1, const float& d)
{
    Vertex temp;
    temp.position = v1.position * d;
    temp.texCoord = v1.texCoord * d;
    for (int i = 0; i < MaxLightMapCount; ++i) {
        temp.lmCoord[i] = v1.lmCoord[i] * d;
    }
    temp.normal = v1.normal * d;
    return temp;
}

void Map::tesselate(int controlOffset, int controlWidth, int vOffset, int iOffset)
{
    Vertex controls[9];
    int cIndex = 0;
    for (int c = 0; c < 3; c++)
    {
        int pos = c * controlWidth;
        controls[cIndex++] = vertexArray[controlOffset + pos];
        controls[cIndex++] = vertexArray[controlOffset + pos + 1];
        controls[cIndex++] = vertexArray[controlOffset + pos + 2];
    }

    int L1 = bezierLevel + 1;

    for (int j = 0; j <= bezierLevel; ++j)
    {
        float a = (float)j / bezierLevel;
        float b = 1.f - a;
        vertexArray[vOffset + j] = controls[0] * b * b + controls[3] * 2 * b * a + controls[6] * a * a;
    }

    for (int i = 1; i <= bezierLevel; ++i)
    {
        float a = (float)i / bezierLevel;
        float b = 1.f - a;

        Vertex temp[3];

        for (int j = 0; j < 3; ++j)
        {
            int k = 3 * j;
            temp[j] = controls[k + 0] * b * b + controls[k + 1] * 2 * b * a + controls[k + 2] * a * a;
        }

        for (int j = 0; j <= bezierLevel; ++j)
        {
            float a = (float)j / bezierLevel;
            float b = 1.f - a;

            vertexArray[vOffset + i * L1 + j] = temp[0] * b * b + temp[1] * 2 * b * a + temp[2] * a * a;
        }
    }

    for (int i = 0; i <= bezierLevel; ++i)
    {
        for (int j = 0; j <= bezierLevel; ++j)
        {
            int offset = iOffset + (i * bezierLevel + j) * 6;
            meshIndexArray[offset + 0] = (i    ) * L1 + (j    ) + vOffset;
            meshIndexArray[offset + 1] = (i    ) * L1 + (j + 1) + vOffset;
            meshIndexArray[offset + 2] = (i + 1) * L1 + (j + 1) + vOffset;

            meshIndexArray[offset + 3] = (i + 1) * L1 + (j + 1) + vOffset;
            meshIndexArray[offset + 4] = (i + 1) * L1 + (j    ) + vOffset;
            meshIndexArray[offset + 5] = (i    ) * L1 + (j    ) + vOffset;
        }
    }
}

#include "shaders.inc"

RenderPass::RenderPass(Map* parent, const glm::vec3& position, const glm::mat4& matrix)
    : pos(position)
    , frutsum(matrix)
{
    renderedFaces.resize(parent->faceArray.size(), false);
}

TracePass::TracePass(Map* parent, const glm::vec3& pos, const glm::vec3 &oldPos, float rad)
    : position(pos)
    , oldPosition(oldPos)
    , radius(rad)
{
    tracedBrushes.resize(parent->brushArray.size(), false);
}

Map::Map()
    : program(0)
    , vertexBuffer(0)
    , meshIndexBuffer(0)
    , bezierLevel(3)
{
    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &meshIndexBuffer);

    GLint status;

    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &vertSrc, NULL);
    glCompileShader(vertShader);
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint length;
        glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &length);
        char* log = new char[length + 1];
        log[length] = '\0';
        glGetShaderInfoLog(vertShader, length, &length, log);
        std::cout << log << std::endl;
        delete[] log;
        return;
    }

    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragSrc, NULL);
    glCompileShader(fragShader);
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint length;
        glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &length);
        char* log = new char[length + 1];
        log[length] = '\0';
        glGetShaderInfoLog(fragShader, length, &length, log);
        std::cout << log << std::endl;
        glDeleteShader(vertShader);
        delete[] log;
        return;
    }

    program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);

    glBindAttribLocation(program, 0, "vertex");
    //glBindAttribLocation(program, 1, "normal");
    glBindAttribLocation(program, 2, "texcoord");
    glBindAttribLocation(program, 3, "lmcoord1");
    glBindAttribLocation(program, 4, "lmcoord2");
    glBindAttribLocation(program, 5, "lmcoord3");
    glBindAttribLocation(program, 6, "lmcoord4");

    glLinkProgram(program);

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        char* log = new char[length + 1];
        log[length] = '\0';
        glGetProgramInfoLog(program, length, &length, log);
        std::cout << log << std::endl;
        delete[] log;
        return;
    }

    programLoc["matrix"] = glGetUniformLocation(program, "matrix");
    programLoc["texture"] = glGetUniformLocation(program, "texture");
    programLoc["lightmap1"] = glGetUniformLocation(program, "lightmap1");
    programLoc["lightmap2"] = glGetUniformLocation(program, "lightmap2");
    programLoc["lightmap3"] = glGetUniformLocation(program, "lightmap3");
    programLoc["lightmap4"] = glGetUniformLocation(program, "lightmap4");
    programLoc["lightmapGamma"] = glGetUniformLocation(program, "lightmapGamma");
}

bool Map::load(std::string filename)
{
    glEnable(GL_TEXTURE_2D);

    PHYSFS_File* file = PHYSFS_openRead(filename.c_str());
    if (!file)
    {
        std::cout << filename.c_str() << ": " << PHYSFS_getLastError() << std::endl;
        return false;
    }

    PHYSFS_seek(file, 0);
    Header header;
    PHYSFS_read(file, &header, sizeof(Header), 1);
    if (std::string(header.magic, 4) == "IBSP")
    {
        // 46 - Quake 3
        // 47 - RTCW
        if (header.version != 0x2E && header.version != 0x2F)
        {
            std::cout << "IBSP version " << header.version << " not supported" << std::endl;
            return false;
        }
        isRBSP = false;
        maxLightMapCount = 1;
        lightMapGamma = 3.0f;
    }
    else if (std::string(header.magic, 4) == "RBSP")
    {
        if (header.version != 1)
        {
            std::cout << "RBSP version " << header.version << " not supported" << std::endl;
            return false;
        }
        isRBSP = true;
        lightMapGamma = 3.0f;
        
        int maxSupportedTextures = 1;
        glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxSupportedTextures);
        maxLightMapCount = std::min<unsigned int>(MaxLightMapCount, maxSupportedTextures);
    }
    else
    {
        std::cout << "Invalid file" << std::endl;
        return false;
    }

    // Read the lumps
    int numLumps = isRBSP ? NUM_LUMPS_RBSP : NUM_LUMPS_IBSP;
    Lump lumps[MAX_LUMPS];
    PHYSFS_read(file, lumps, sizeof(Lump), numLumps);

    PHYSFS_seek(file, lumps[ENTITY].offset);
    std::string rawEntity;
    rawEntity.resize(lumps[ENTITY].size);
    PHYSFS_read(file, &rawEntity[0], 1, lumps[ENTITY].size);

    int shaderCount = lumps[SHADER].size / sizeof(RawShader);
    PHYSFS_seek(file, lumps[SHADER].offset);
    shaderArray.reserve(shaderCount);
    for (int i = 0; i < shaderCount; i++)
    {
        RawShader rawshader;
        PHYSFS_read(file, &rawshader, sizeof(RawShader), 1);
        rawshader.name[63] = '\0';
        Shader shader;
        shader.render = true;
        shader.transparent = false;
        shader.solid = true;
        shader.name = std::string(rawshader.name);
        if (rawshader.surface & SURF_NONSOLID) shader.solid = false;
        if (rawshader.contents & CONTENTS_PLAYERCLIP) shader.solid = true;
        if (rawshader.contents & CONTENTS_TRANSLUCENT) shader.transparent = true;
        if (rawshader.contents & CONTENTS_LAVA) shader.render = false;
        if (rawshader.contents & CONTENTS_SLIME) shader.render = false;
        if (rawshader.contents & CONTENTS_WATER) shader.render = false;
        if (rawshader.contents & CONTENTS_FOG) shader.render = false;
        if (shader.name == "noshader") shader.render = false;
        if (shader.render)
        {
            if (PHYSFS_exists(std::string(shader.name + ".jpg").c_str()))
            {
                shader.name += ".jpg";
            }
            else if (PHYSFS_exists(std::string(shader.name + ".tga").c_str()))
            {
                shader.name += ".tga";
            }
            if ((rawshader.surface & SURF_NODRAW) == 0)
            {
                bool texOK = false;
                FileStream filestream(shader.name);
                if (filestream.isOpen())
                {
                    if (shader.texture.loadFromStream(filestream))
                    {
#if SFML_VERSION_MAJOR > 2 || (SFML_VERSION_MAJOR == 2 && SFML_VERSION_MINOR >= 4)
                        shader.texture.generateMipmap();
#endif
                        shader.texture.setRepeated(true);
                        shader.texture.setSmooth(true);
                        texOK = true;
                    }
                }
                if (!texOK)
                {
                    // set an error texture
                    sf::Image image;
                    image.create(1, 1, sf::Color(255, 0, 255));
                    shader.texture.loadFromImage(image);

                    std::cout << shader.name << ": Texture not found" << std::endl;
                }
            }
        }
        shaderArray.push_back(shader);
    }

    int planeCount = lumps[PLANE].size / sizeof(Plane);
    PHYSFS_seek(file, lumps[PLANE].offset);
    assert(planeCount * sizeof(Plane) == lumps[PLANE].size);
    planeArray.resize(planeCount);
    if (planeCount > 0)
        PHYSFS_read(file, &planeArray[0], sizeof(Plane), planeCount);

    int nodeCount = lumps[NODE].size / sizeof(Node);
    assert(nodeCount * sizeof(Node) == lumps[NODE].size);
    PHYSFS_seek(file, lumps[NODE].offset);
    nodeArray.resize(nodeCount);
    if (nodeCount > 0)
        PHYSFS_read(file, &nodeArray[0], sizeof(Node), nodeCount);

    int leafCount = lumps[LEAF].size / sizeof(Leaf);
    assert(leafCount * sizeof(Leaf) == lumps[LEAF].size);
    PHYSFS_seek(file, lumps[LEAF].offset);
    leafArray.resize(leafCount);
    if (leafCount > 0)
        PHYSFS_read(file, &leafArray[0], sizeof(Leaf), leafCount);

    int leafFaceCount = lumps[LEAFFACE].size / sizeof(int);
    assert(leafFaceCount * sizeof(int) == lumps[LEAFFACE].size);
    PHYSFS_seek(file, lumps[LEAFFACE].offset);
    leafFaceArray.resize(leafFaceCount);
    if (leafFaceCount > 0)
        PHYSFS_read(file, &leafFaceArray[0], sizeof(int), leafFaceCount);

    int leafBrushCount = lumps[LEAFBRUSH].size / sizeof(int);
    assert(leafBrushCount * sizeof(int) == lumps[LEAFBRUSH].size);
    PHYSFS_seek(file, lumps[LEAFBRUSH].offset);
    leafBrushArray.resize(leafBrushCount);
    if (leafBrushCount > 0)
        PHYSFS_read(file, &leafBrushArray[0], sizeof(int), leafBrushCount);

    int modelCount = lumps[MODEL].size / sizeof(Model);
    assert(modelCount * sizeof(Model) == lumps[MODEL].size);
    PHYSFS_seek(file, lumps[MODEL].offset);
    modelArray.resize(modelCount);
    if (modelCount > 0)
        PHYSFS_read(file, &modelArray[0], sizeof(Model), modelCount);

    int brushCount = lumps[BRUSH].size / sizeof(Brush);
    assert(brushCount * sizeof(Brush) == lumps[BRUSH].size);
    PHYSFS_seek(file, lumps[BRUSH].offset);
    brushArray.resize(brushCount);
    if (brushCount > 0)
        PHYSFS_read(file, &brushArray[0], sizeof(Brush), brushCount);

    PHYSFS_seek(file, lumps[BRUSHSIDE].offset);
    if (isRBSP) {
        int brushSideCount = lumps[BRUSHSIDE].size / sizeof(BrushSide);
        assert(brushSideCount * sizeof(BrushSide) == lumps[BRUSHSIDE].size);
        brushSideArray.resize(brushSideCount);
        if (brushSideCount > 0)
            PHYSFS_read(file, &brushSideArray[0], sizeof(BrushSide), brushSideCount);
    } else {
        int brushSideCount = lumps[BRUSHSIDE].size / sizeof(RawBrushSide_IBSP);
        assert(brushSideCount * sizeof(RawBrushSide_IBSP) == lumps[BRUSHSIDE].size);
        brushSideArray.resize(brushSideCount);
        if (brushSideCount > 0)
        {
            std::vector<RawBrushSide_IBSP> rawBrushSideArray(brushSideCount);
            PHYSFS_read(file, &rawBrushSideArray[0], sizeof(RawBrushSide_IBSP), brushSideCount);
            for (int i = 0; i < brushSideCount; ++i) {
                brushSideArray[i] = ibsp_to_rbsp(rawBrushSideArray[i]);
            }
        }
    }

    int effectCount = lumps[EFFECT].size / sizeof(Effect);
    assert(effectCount * sizeof(Effect) == lumps[EFFECT].size);
    PHYSFS_seek(file, lumps[EFFECT].offset);
    effectArray.resize(effectCount);
    if (effectCount > 0)
        PHYSFS_read(file, &effectArray[0], sizeof(Effect), effectCount);

    constexpr int lightMapSize = (128 * 128 * 3);
    int lightMapCount = lumps[LIGHTMAP].size / lightMapSize;
    assert(lightMapCount * lightMapSize == lumps[LIGHTMAP].size);
    PHYSFS_seek(file, lumps[LIGHTMAP].offset);
    lightMapArray.resize(lightMapCount + 3);
    for (int i = 0; i < lightMapCount; i++)
    {
        std::array<sf::Uint8, 128 * 128 * 4> rawLightMap;
        for (int i = 0; i < 128 * 128; i++)
        {
            PHYSFS_read(file, &rawLightMap[i * 4], 3, 1);
            rawLightMap[i * 4 + 3] = 255;
        }
        sf::Image image;
        image.create(128, 128, &rawLightMap[0]);
        sf::Texture &texture = lightMapArray[i];
        texture.loadFromImage(image);
        texture.setRepeated(true);
        texture.setSmooth(true);
    }
    {
        sf::Image image;
        image.create(1, 1, sf::Color(85, 85, 85));
        lightMapArray[lightMapCount].loadFromImage(image);
    }
    {
        sf::Image image;
        image.create(1, 1, sf::Color(0, 0, 0));
        lightMapArray[lightMapCount+1].loadFromImage(image);
    }
    {
        sf::Image image;
        image.create(1, 1, sf::Color(255,255,255));
        lightMapArray[lightMapCount+1].loadFromImage(image);
    }

    int defaultLightMap = lightMapCount;
    int defaultDarkMap = lightMapCount + 1;
    int defaultWhiteMap = lightMapCount + 2;

    int faceCount;
    std::vector<RawFace_RBSP> rawRbspFaces;
    PHYSFS_seek(file, lumps[FACE].offset);
    if (isRBSP) {
        faceCount = lumps[FACE].size / sizeof(RawFace_RBSP);
        assert(faceCount * sizeof(RawFace_RBSP) == lumps[FACE].size);
        rawRbspFaces.resize(faceCount);
        if (faceCount > 0) {
            PHYSFS_read(file, &rawRbspFaces[0], sizeof(RawFace_RBSP), faceCount);
        }
    } else {
        faceCount = lumps[FACE].size / sizeof(RawFace_IBSP);
        assert(faceCount * sizeof(RawFace_IBSP) == lumps[FACE].size);
        rawRbspFaces.resize(faceCount);
        if (faceCount > 0) {
            std::vector<RawFace_IBSP> rawIbspFaces(faceCount);
            PHYSFS_read(file, &rawIbspFaces[0], sizeof(RawFace_IBSP), faceCount);
            for (int i = 0; i < faceCount; ++i) {
                rawRbspFaces[i] = ibsp_to_rbsp(rawIbspFaces[i]);
            }
        }
    }

    int bezierCount = 0;
    int bezierPatchSize = (bezierLevel + 1) * (bezierLevel + 1);
    int bezierIndexSize = bezierLevel * bezierLevel * 6;
    faceArray.resize(rawRbspFaces.size());
    for (int i = 0; i < faceCount; i++)
    {
        const RawFace_RBSP& rawFace = rawRbspFaces[i];
        Face &face = faceArray[i];
        face.shader = rawFace.shader;
        face.effect = rawFace.effect;
        face.vertexOffset = rawFace.vertexOffset;
        face.vertexCount = rawFace.vertexCount;
        face.meshIndexOffset = rawFace.meshVertexOffset;
        face.meshIndexCount = rawFace.meshVertexCount;
        copy_array(face.lightMaps, rawFace.lightMaps);
        for (int i = 0; i < MaxLightMapCount; ++i) {
            int lmIndex = rawFace.lightMaps[i];
            switch (rawFace.lightMaps[i]) {
            case LIGHTMAP_NONE:
                face.lightMaps[i] = defaultDarkMap;
                break;
            case LIGHTMAP_WHITEIMAGE:
                face.lightMaps[i] = defaultWhiteMap;
                break;
            case LIGHTMAP_BY_VERTEX:
                // todo: implement someday
                face.lightMaps[i] = defaultLightMap;
                break;
            default:
                if (lmIndex < 0) {
                    face.lightMaps[i] = defaultDarkMap;
                } else {
                    face.lightMaps[i] = lmIndex;
                }
            }
        }
        switch (rawFace.type)
        {
        case 1:
            face.type = Face::Brush;
            break;
        case 2:
            face.type = Face::Bezier;
            break;
        case 3:
            face.type = Face::Model;
            break;
        default:
            face.type = Face::None;
            break;
        }

        if (face.type == Face::Bezier)
        {
            face.bezierSize[0] = rawFace.size[0];
            face.bezierSize[1] = rawFace.size[1];
            int dimX = (face.bezierSize[0] - 1) / 2;
            int dimY = (face.bezierSize[0] - 1) / 2;
            int size = dimX * dimY;
            bezierCount += size;
        }
    }

    int meshVertexCount = lumps[MESHVERTEX].size / sizeof(GLuint);
    assert(meshVertexCount * sizeof(GLuint) == lumps[MESHVERTEX].size);
    PHYSFS_seek(file, lumps[MESHVERTEX].offset);
    meshIndexArray.resize(meshVertexCount + bezierIndexSize * bezierCount);
    if (meshVertexCount > 0)
        PHYSFS_read(file, &meshIndexArray[0], sizeof(GLuint), meshVertexCount);

    int vertexCount;
    PHYSFS_seek(file, lumps[VERTEX].offset);
    if (isRBSP) {
        vertexCount = lumps[VERTEX].size / sizeof(Vertex);
        assert(vertexCount * sizeof(Vertex) == lumps[VERTEX].size);
        vertexArray.resize(vertexCount);
        if (vertexCount > 0)
            PHYSFS_read(file, &vertexArray[0], sizeof(Vertex), vertexCount);
    } else {
        vertexCount = lumps[VERTEX].size / sizeof(RawVertex_IBSP);
        assert(vertexCount * sizeof(RawVertex_IBSP) == lumps[VERTEX].size);
        vertexArray.resize(vertexCount);
        if (vertexCount > 0) {
            std::vector<RawVertex_IBSP> rawVertexArray(vertexCount);
            PHYSFS_read(file, &rawVertexArray[0], sizeof(RawVertex_IBSP), vertexCount);
            for (int i = 0; i < vertexCount; ++i) {
                vertexArray[i] = ibsp_to_rbsp(rawVertexArray[i]);
            }
        }
    }

    vertexArray.resize(vertexCount + bezierCount * bezierPatchSize);
    for (int i = 0, vOffset = vertexCount, iOffset = meshVertexCount; i < faceCount; i++)
    {
        Face &face = faceArray[i];
        if (face.type == Face::Bezier)
        {
            int dimX = (face.bezierSize[0] - 1) / 2;
            int dimY = (face.bezierSize[1] - 1) / 2;

            face.meshIndexOffset = iOffset;

            for (int x = 0, n = 0; n < dimX; n++, x = 2 * n)
            {
                for (int y = 0, m = 0; m < dimY; m++, y = 2 * m)
                {
                    tesselate(face.vertexOffset + x + face.bezierSize[0] * y, face.bezierSize[0], vOffset, iOffset);
                    vOffset += bezierPatchSize;
                    iOffset += bezierIndexSize;
                }
            }

            face.meshIndexCount = iOffset - face.meshIndexOffset;
        }
        else
        {
            for (int i = 0; i < face.meshIndexCount; i++)
            {
                meshIndexArray[face.meshIndexOffset + i] += face.vertexOffset;
            }
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vertexArray.size() * sizeof(Vertex), &vertexArray[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, meshIndexArray.size() * sizeof(GLuint), &meshIndexArray[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    int lightVolCount;
    PHYSFS_seek(file, lumps[LIGHTVOL].offset);
    std::vector<RawLightVol_RBSP> rawRbspLightVolArray;
    if (isRBSP) {
        lightVolCount = lumps[LIGHTVOL].size / sizeof(RawLightVol_RBSP);
        assert(lightVolCount * sizeof(RawLightVol_RBSP) == lumps[LIGHTVOL].size);
        rawRbspLightVolArray.resize(lightVolCount);
        if (lightVolCount > 0) {
            PHYSFS_read(file, &rawRbspLightVolArray[0], sizeof(RawLightVol_RBSP), lightVolCount);
        }
    } else {
        lightVolCount = lumps[LIGHTVOL].size / sizeof(RawLightVol_IBSP);
        assert(lightVolCount * sizeof(RawLightVol_IBSP) == lumps[LIGHTVOL].size);
        rawRbspLightVolArray.resize(lightVolCount);
        if (lightVolCount > 0) {
            std::vector<RawLightVol_IBSP> rawIbspLightVolArray(lightVolCount);
            PHYSFS_read(file, &rawIbspLightVolArray[0], sizeof(RawLightVol_IBSP), lightVolCount);
            for (int i = 0; i < lightVolCount; ++i) {
                rawRbspLightVolArray[i] = ibsp_to_rbsp(rawIbspLightVolArray[i]);
            }
        }
    }

    lightVolArray.reserve(lightVolCount);
    for (int i = 0; i < lightVolCount; i++)
    {
        const auto& rawLightVol = rawRbspLightVolArray[i];
        LightVol lightVol;

        for (int lmIndex = 0; lmIndex < MaxLightMapCount; ++lmIndex) {
            lightVol.ambient[lmIndex].x = rawLightVol.ambient[lmIndex][0];
            lightVol.ambient[lmIndex].y = rawLightVol.ambient[lmIndex][1];
            lightVol.ambient[lmIndex].z = rawLightVol.ambient[lmIndex][2];
            lightVol.ambient[lmIndex] = lightVol.ambient[lmIndex] / 256.f;

            lightVol.directional[lmIndex].x = rawLightVol.directional[lmIndex][0];
            lightVol.directional[lmIndex].y = rawLightVol.directional[lmIndex][1];
            lightVol.directional[lmIndex].z = rawLightVol.directional[lmIndex][2];
            lightVol.directional[lmIndex] = lightVol.directional[lmIndex] / 256.f;
        }

        float phi = (int(rawLightVol.direction[0]) - 128) / 256.f * 180;
        float thetha = int(rawLightVol.direction[1]) / 256.f * 360;

        lightVol.direction.x = sin(thetha) * cos(phi);
        lightVol.direction.y = cos(thetha) * cos(phi);
        lightVol.direction.z = sin(phi);
        lightVol.direction = glm::normalize(lightVol.direction);

        lightVolArray.push_back(lightVol);
    }

    PHYSFS_seek(file, lumps[VISDATA].offset);
    if (lumps[VISDATA].size > 0)
    {
        PHYSFS_read(file, &visData.clusterCount, sizeof(int), 1);
        PHYSFS_read(file, &visData.bytesPerCluster, sizeof(int), 1);
        unsigned int size = visData.clusterCount * visData.bytesPerCluster;

        std::vector<char> rawVisData(size);
        PHYSFS_read(file, &rawVisData[0], size, 1);

        visData.data.resize(size * 8, false);
        for (unsigned long int byteIndex = 0; byteIndex < size; byteIndex++)
        {
            unsigned char byte = rawVisData.at(byteIndex);
            for (unsigned int bit = 0; bit < 8; bit++)
            {
                if (byte & (1 << bit))
                    visData.data[byteIndex * 8 + bit] = true;
            }
        }
    }

    if (isRBSP) {
        PHYSFS_seek(file, lumps[LIGHTARRAY].offset);
        int lightGridCount = lumps[LIGHTARRAY].size / sizeof(unsigned short);
        assert(lightGridCount * sizeof(unsigned short) == lumps[LIGHTARRAY].size);
        // todo: LIGHTARRAY lump
    }

    PHYSFS_close(file);

    lightVolSizeX = int(floor(modelArray[0].max.x / 64) - ceil(modelArray[0].min.x / 64) + 1);
    lightVolSizeY = int(floor(modelArray[0].max.y / 64) - ceil(modelArray[0].min.y / 64) + 1);
    lightVolSizeZ = int(floor(modelArray[0].max.z / 128) - ceil(modelArray[0].min.z / 128) + 1);

    glDisable(GL_TEXTURE_2D);
    return true;
}

bool Map::clusterVisible(int test, int cam)
{
    if (visData.data.size() == 0 || cam < 0 || test < 0)
        return true;

    return visData.data[test * visData.bytesPerCluster * 8 + cam];
}

int Map::findLeaf(glm::vec3& pos)
{
    int index = 0;
    while (index >= 0)
    {
        Node& node = nodeArray[index];
        Plane& plane = planeArray[node.plane];
        if (glm::dot(plane.normal, pos) >= plane.distance)
        {
            index = node.children[0];
        }
        else
        {
            index = node.children[1];
        }
    };
    return ~index;
}

LightVol Map::findLightVol(glm::vec3& pos)
{
    if (lightVolArray.size() == 0)
        return LightVol();
    int cellX = int(floor(pos.x / 64) - ceil(modelArray[0].min.x / 64));
    int cellY = int(floor(pos.y / 64) - ceil(modelArray[0].min.y / 64));
    int cellZ = int(floor(pos.z / 128) - ceil(modelArray[0].min.z / 128));
    cellX = std::min(std::max(cellX, 0), (int)lightVolSizeX);
    cellY = std::min(std::max(cellY, 0), (int)lightVolSizeY);
    cellZ = std::min(std::max(cellZ, 0), (int)lightVolSizeZ);
    unsigned int index = cellX;
    index += cellY * lightVolSizeX;
    index += cellZ * lightVolSizeX * lightVolSizeY;
    return lightVolArray[index];
}

void Map::renderFace(int index, RenderPass& pass, bool solid)
{
    if (pass.renderedFaces[index])
        return;
    Face& face = faceArray[index];
    if (shaderArray[face.shader].transparent == solid)
        return;
    if (!shaderArray[face.shader].render)
        return;

    glActiveTexture(GL_TEXTURE0);
    sf::Texture::bind(&shaderArray[face.shader].texture);

    for (int i = 0; i < maxLightMapCount; ++i) {
        glActiveTexture(GL_TEXTURE1 + i);
        sf::Texture::bind(&lightMapArray[face.lightMaps[i]]);
    }
    
    glDrawElements(GL_TRIANGLES, face.meshIndexCount, GL_UNSIGNED_INT, (void*)(long)(face.meshIndexOffset * sizeof(GLuint)));

    pass.renderedFaces[index] = true;
}

void Map::renderNode(int index, RenderPass& pass, bool solid)
{
    if (index < 0)
    {
        Leaf& leaf = leafArray[~index];
        if (!clusterVisible(leaf.cluster, pass.cluster))
            return;
        if (!pass.frutsum.insideAABB(leaf.max, leaf.min))
            return;

        for (int i = 0; i < leaf.faceCount; i++)
        {
            int faceIndex = leafFaceArray[i + leaf.faceOffset];
            renderFace(faceIndex, pass, solid);
        }
        return;
    }

    Node& node = nodeArray[index];
    if (!pass.frutsum.insideAABB(node.max, node.min))
        return;

    Plane& plane = planeArray[node.plane];

    if ((glm::dot(plane.normal, pass.pos) >= plane.distance) == solid)
    {
        renderNode(node.children[0], pass, solid);
        renderNode(node.children[1], pass, solid);
    }
    else
    {
        renderNode(node.children[1], pass, solid);
        renderNode(node.children[0], pass, solid);
    }
}

void Map::renderWorld(glm::mat4 matrix, glm::vec3 pos)
{
    glFrontFace(GL_CW);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIndexBuffer);

    if (nodeArray.size() == 0)
        return;

    glUseProgram(program);
    glEnableVertexAttribArray(0);
    //glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
    glEnableVertexAttribArray(5);
    glEnableVertexAttribArray(6);
    glUniformMatrix4fv(programLoc["matrix"], 1, GL_FALSE, &matrix[0][0]);
    glUniform1i(programLoc["texture"], 0);
    glUniform1i(programLoc["lightmap1"], 1);
    glUniform1i(programLoc["lightmap2"], 2);
    glUniform1i(programLoc["lightmap3"], 3);
    glUniform1i(programLoc["lightmap4"], 4);
    glUniform1f(programLoc["lightmapGamma"], lightMapGamma);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), VertexPosition);
    //glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE,  sizeof(Vertex), VertexNormal);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), VertexTexCoord);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), VertexLMCoord0);
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), VertexLMCoord1);
    glVertexAttribPointer(5, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), VertexLMCoord2);
    glVertexAttribPointer(6, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), VertexLMCoord3);

    RenderPass pass(this, pos, matrix);
    pass.cluster = leafArray[findLeaf(pos)].cluster;

    glEnable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    renderNode(0, pass, true);

    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    renderNode(0, pass, false);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
}

void Map::traceBrush(int index, TracePass& pass)
{
    if (pass.tracedBrushes[index])
        return;
    pass.tracedBrushes[index] = true;
    Brush& brush = brushArray[index];
    if (!shaderArray[brush.shader].solid)
        return;

    Plane* collidingPlane = NULL;
    float collidingDist = 0.0;

    for (int i = 0; i < brush.sideCount; i++)
    {
        BrushSide& side = brushSideArray[i + brush.sideOffset];
        Plane& plane = planeArray[side.plane];

        if (glm::dot(plane.normal, pass.oldPosition) - plane.distance < pass.radius)
            continue;

        float dist = glm::dot(plane.normal, pass.position) - plane.distance - pass.radius;

        if (dist > 0.f)
            return;

        if (!shaderArray[side.shader].solid)
            continue;

        if (collidingPlane == NULL || dist > collidingDist)
        {
            collidingPlane = &plane;
            collidingDist = dist;
        }
    }

    if (collidingPlane == NULL)
        return;
    pass.position -= collidingPlane->normal * collidingDist;
}

void Map::traceNode(int index, TracePass& pass)
{
    if (index < 0)
    {
        Leaf& leaf = leafArray[~index];
        for (int i = 0; i < leaf.brushCount; i++)
        {
            traceBrush(leafBrushArray[i + leaf.brushOffset], pass);
        }
        return;
    }

    Node& node = nodeArray[index];
    Plane& plane = planeArray[node.plane];
    float dist = glm::dot(plane.normal, pass.position) - plane.distance;

    if (dist > -pass.radius)
    {
        traceNode(node.children[0], pass);
    }

    if (dist < pass.radius)
    {
        traceNode(node.children[1], pass);
    }
}

glm::vec3 Map::traceWorld(glm::vec3 pos, glm::vec3 oldPos, float radius)
{
    TracePass pass(this, pos, oldPos, radius);
    traceNode(0, pass);

    return pass.position;
}
