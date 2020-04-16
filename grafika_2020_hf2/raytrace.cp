//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

enum MaterialType { ROUGH, REFLECTIVE };

struct Material {
    vec3 ka, kd, ks;
    float  shininess;
    vec3 F0;
    MaterialType type;
    Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
        ka = _kd * M_PI;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
    }
};

vec3 operator/(vec3 num, vec3 denom) {
    return(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
    ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
        vec3 one(1, 1, 1);
        F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
    }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material* material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
    Material* material;
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

class Sphere : public Intersectable {
    vec3 center;
    float radius;
public:
    Sphere(const vec3& _center, float _radius, Material* _material) {
        center = _center;
        radius = _radius;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(ray.dir, ray.dir);
        float b = dot(dist, ray.dir) * 2.0f;
        float c = dot(dist, dist) - radius * radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - center) * (1.0f / radius);
        hit.material = material;
        return hit;
    }
};

class Room : public Intersectable {
    vec3 center;
    vec4 r;
public:
    Room(const vec3& _center, vec4 _r, Material* _material) {
        center = _center;
        material = _material;
        r = _r;
    }

    mat4 Q() {
        return mat4(
            vec4(1 / (r.x * r.x), 0, 0, 0),
            vec4(0, 1 / (r.y * r.y), 0, 0),
            vec4(0, 0, 1 / (r.z * r.z), 0),
            vec4(0, 0, 0, -1)
        );
    }

    vec3 gradf(vec4 r) {
        vec4 g = r * Q() * 2;
        return vec3(g.x, g.y, g.z);
    }


    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * Q(), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float b = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * Q(), vec4(ray.start.x, ray.start.y, ray.start.z, 1)) + dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * Q(), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float c = dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * Q(), vec4(ray.start.x, ray.start.y, ray.start.z, 1));
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;

        hit.position = ray.start + ray.dir * hit.t;
        hit.material = material;
        if (hit.position.z  > 1.1f) {
            hit.t = t1;
            hit.position = ray.start + ray.dir * hit.t;
//            vec3 kd1(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
//            Material* material1 = new RoughMaterial(kd1, ks, 50);
            hit.material = material;
            if (hit.position.z > 1.1f) {
                Hit hit1;
                hit1.t = -1;
                return hit1;
            }
        }
        hit.normal = gradf(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        return hit;
    }
};


class Ellipsoid : public Intersectable {
    vec3 center, s, t;
    vec4 r;
public:
    Ellipsoid(const vec3& _center, vec4 _r, vec3 _s, vec3 _t, Material* _material) {
        center = _center;
        material = _material;
        r = _r;
        s = _s;
        t = _t;
    }

    mat4 Q() {
        return mat4(
            vec4(1 / (r.x * r.x), 0, 0, 0),
            vec4(0, 1 / (r.y * r.y), 0, 0),
            vec4(0, 0, 1 / (r.z * r.z), 0),
            vec4(0, 0, 0, -1)
        );
    }
    
    mat4 T() {
        return mat4(vec4(s.x, 0, 0, 0),
            vec4(0, s.y, 0, 0),
            vec4(0, 0, s.z, 0),
            vec4(t.x, t.y, t.z, 1));
    }
    
    mat4 TI() {
        return mat4(vec4(1 / s.x, 0, 0, 0),
            vec4(0, 1 / s.y, 0, 0),
            vec4(0, 0, 1 / s.z, 0),
            vec4(-(t.x / s.x), -(t.y / s.y), -(t.z / s.z), 1));
    }
    
    mat4 TIT() {
        return mat4(
                    vec4(1/s.x, 0, 0, -(t.x / s.x)),
                    vec4(0, 1/s.y, 0, -(t.y/s.y)),
                    vec4(0, 0, 1/s.z, -(t.z/s.z)),
                    vec4(0, 0, 0, 1)
                    );
    }

    vec3 gradf(vec4 r) {
        vec4 g = r * TI() * Q() * TIT() * 2;
        return vec3(g.x, g.y, g.z);
    }


    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TI() * Q() * TIT()), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float b = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TI() * Q() * TIT()), vec4(ray.start.x, ray.start.y, ray.start.z, 1)) + dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TI() * Q() * TIT()), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float c = dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TI() * Q() * TIT()), vec4(ray.start.x, ray.start.y, ray.start.z, 1));
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;

        hit.position = ray.start + ray.dir * hit.t;
        hit.material = material;
        hit.t = t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = gradf(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        
        return hit;
    }
};


class Camera {
    vec3 eye, lookat, right, up;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        //        float focus = length(w);
        //        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        //        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
        float windowSize = length(w) * tanf(fov / 2);
        right = normalize(cross(vup, w)) * windowSize;
        up = normalize(cross(w, right)) * windowSize;
    }
    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }
};

struct Light {
    vec3 direction;
    vec3 Le;
    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
    std::vector<Intersectable*> objects;
    std::vector<Light*> lights;
    Camera camera;
    vec3 La;
public:
    void build() {
//        vec3 eye = vec3(0, -4.0f, 0), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0.1f);
        vec3 eye = vec3(0, -6.0f, 0), vup = vec3(0, 0, 1), lookat = vec3(0, 0, -1.0f);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.5f, 0.5f, 0.5f);
        vec3 lightDirection(-0.2f, 0, 1), Le(2, 2, 2);
        lights.push_back(new Light(lightDirection, Le));

        //vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f), ks(2, 2, 2);


        //vec3 n(0.17f, 0.35f, 1.5f);
        //vec3 kappa(3.1f, 2.7f, 1.9f);
        //Material* material2 = new ReflectiveMaterial(n, kappa);
        //for (int i = 0; i < 150; i++) {
            //objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, material1));
            //objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, material2));
        //}

        vec3 kd1(0.3f, 0.2f, 0.1f);
        vec3 kd2(0.1f, 0.2f, 0.3f);
        vec3 ks(2, 2, 2);
        vec3 grey(0.4f, 0.2f, 0.4f);
        Material* blueRough = new RoughMaterial(kd2, ks, 50);
        Material* orangeRough = new RoughMaterial(kd1, ks, 50);
        Material* greyRough = new RoughMaterial(grey, ks, 50);

        objects.push_back(new Room(vec3(0, 0, 0), vec4(2.4f, 6, 1.4f), blueRough));
        objects.push_back(new Ellipsoid(vec3(0, 0, 0), vec4(0.4f, 1.5f, 1), vec3(0.7f,0.7f,0.7f), vec3(-0.6f, -1, -0.6f), greyRough));

    }

    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable* object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {    // for directional lights
        for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        if (depth > 5) return La;
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;

        vec3 outRadiance(0, 0, 0);
        if (hit.material->type == ROUGH) {
            outRadiance = hit.material->ka * La;
            for (Light* light : lights) {
                Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
                float cosTheta = dot(hit.normal, light->direction);
                if (cosTheta > 0 && !shadowIntersect(shadowRay)) {    // shadow computation
                    outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                    vec3 halfway = normalize(-ray.dir + light->direction);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
                }
            }
        }

        if (hit.material->type == REFLECTIVE) {
            vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
            float cosa = -dot(ray.dir, hit.normal);
            vec3 one(1, 1, 1);
            vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
            outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
        }

        return outRadiance;
    }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
    #version 330
    precision highp float;
    layout(location = 0) in vec2 cVertexPosition;    // Attrib Array 0
    out vec2 texcoord;
    void main() {
        texcoord = (cVertexPosition + vec2(1, 1))/2;                            // -1,1 to 0,1
        gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);         // transform to clipping space
    }
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
    #version 330
    precision highp float;
    uniform sampler2D textureUnit;
    in  vec2 texcoord;            // interpolated texture coordinates
    out vec4 fragmentColor;        // output that goes to the raster memory as told by glBindFragDataLocation
    void main() {
        fragmentColor = texture(textureUnit, texcoord);
    }
)";

class FullScreenTexturedQuad {
    unsigned int vao = 0, textureId = 0;    // vertex array object id and texture id
//    Texture texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight)
        //        : texture(windowWidth, windowHeight)
    {
        glGenVertexArrays(1, &vao);    // create 1 vertex array object
        glBindVertexArray(vao);        // make it active

        unsigned int vbo;        // vertex buffer objects
        glGenBuffers(1, &vbo);    // Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };    // two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);       // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    void LoadTexture(std::vector<vec4>& image) {
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
    }

    void Draw() {
        glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
        int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
        const unsigned int textureUnit = 0;
        if (location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);    // draw two triangles forming a quad
    }
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    //    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    //    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    //    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

        // copy image to GPU as a texture
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    fullScreenTexturedQuad->LoadTexture(image);
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();                                    // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    glutPostRedisplay();
}
