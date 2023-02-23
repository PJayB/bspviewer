#ifndef BSP_HPP
#define BSP_HPP

#include <string>
#include <vector>
#include <map>
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <SFML/Graphics/Texture.hpp>
#include "frutsum.hpp"

class Map;

// RBSP lightmap count
constexpr unsigned int MaxLightMapCount = 4;

struct Plane {
    glm::vec3 normal;
    float distance;
};

struct Node {
    int plane;
    int children[2];
    int min[3];
    int max[3];
};

struct Leaf {
    int cluster;
    int area;
    int min[3];
    int max[3];
    int faceOffset;
    int faceCount;
    int brushOffset;
    int brushCount;
};

struct Model {
    glm::vec3 min;
    glm::vec3 max;
    int faceOffset;
    int faceCount;
    int brushOffset;
    int brushCount;
};

struct Brush {
    int sideOffset;
    int sideCount;
    int shader;
};

struct BrushSide {
    int plane;
    int shader;
    int drawSurf;
};

struct Vertex {
    glm::vec3 position;
    glm::vec2 texCoord;
    glm::vec2 lmCoord[MaxLightMapCount];
    glm::vec3 normal;
    unsigned char colour[MaxLightMapCount][4];
};

struct Effect {
    char name[64];
    int brush;
    int unknown;
};

struct Face {
    enum Type
    {
        Brush,
        Model,
        Bezier,
        None
    };

    Type type;
    int shader;
    int effect;
    int vertexOffset;
    int vertexCount;
    int meshIndexOffset;
    int meshIndexCount;
    int lightMaps[MaxLightMapCount];
    int bezierSize[2];
};

struct LightVol {
    glm::vec3 ambient[MaxLightMapCount];
    glm::vec3 directional[MaxLightMapCount];
    unsigned char styles[MaxLightMapCount]; // todo: find definition of what this does
    glm::vec3 direction;
};

struct VisData {
    int clusterCount;
    int bytesPerCluster;
    std::vector<bool> data;
};

struct Shader {
    bool transparent;
    bool render;
    bool solid;
    std::string name;
    sf::Texture texture;
};

struct RenderPass {
    glm::vec3 pos;
    Frutsum frutsum;

    int cluster;
    std::vector<bool> renderedFaces;

    RenderPass(Map* parent, const glm::vec3 &position, const glm::mat4 &matrix);
};

struct TracePass {
    glm::vec3 position;
    glm::vec3 oldPosition;
    float radius;

    std::vector<bool> tracedBrushes;

    TracePass(Map* parent, const glm::vec3 &pos, const glm::vec3 &oldPos, float rad);
};

class Map
{
protected:
    GLuint program;
    GLuint vertexBuffer;
    GLuint meshIndexBuffer;
    std::map<std::string, GLuint> programLoc;
    VisData visData;
    int bezierLevel;
    bool isRBSP;

    std::vector<Plane> planeArray;
    std::vector<Node> nodeArray;
    std::vector<Leaf> leafArray;
    std::vector<int> leafFaceArray;
    std::vector<int> leafBrushArray;
    std::vector<Model> modelArray;
    std::vector<Brush> brushArray;
    std::vector<BrushSide> brushSideArray;
    std::vector<Vertex> vertexArray;
    std::vector<GLuint> meshIndexArray;
    std::vector<Effect> effectArray;
    std::vector<Face> faceArray;
    std::vector<sf::Texture> lightMapArray;
    std::vector<LightVol> lightVolArray;
    std::vector<Shader> shaderArray;

    unsigned int lightVolSizeX;
    unsigned int lightVolSizeY;
    unsigned int lightVolSizeZ;
    unsigned int maxLightMapCount;
    float lightMapGamma;

    void tesselate(int controlOffset, int controlWidth, int vOffset, int iOffset);

    bool clusterVisible(int test, int cam);
    int findLeaf(glm::vec3 &pos);
    int findLeafCluster(glm::vec3 &pos);
    LightVol findLightVol(glm::vec3 &pos);

    void drawMesh(int faceIndex);
    void drawPatch(int faceIndex);

    void renderFace(int index, RenderPass &pass, bool solid);
    void renderNode(int index, RenderPass &pass, bool solid);

    void traceBrush(int index, TracePass &pass);
    void traceNode(int index, TracePass &pass);

public:
    Map();

    bool load(std::string fileName);
    void renderWorld(glm::mat4 matrix, glm::vec3 pos);
    glm::vec3 traceWorld(glm::vec3 pos, glm::vec3 oldPos, float radius);

    friend struct Bezier;
    friend struct Patch;
    friend struct RenderPass;
    friend struct TracePass;
};

#endif // BSP_HPP
