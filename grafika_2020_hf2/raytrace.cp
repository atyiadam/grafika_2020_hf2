//=============================================================================================
// Computer Graphics
//=============================================================================================
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Atyi Adam
// Neptun : PTZY5J
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
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
    return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
    ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
        vec3 one(1.0f, 1.0f, 1.0f);
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
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - center) * (1.0f / radius);
        hit.material = material;
        return hit;
    }
};

class Hiperboloid : public Intersectable {
    vec3 center;
    vec4 r;
    vec3 s;
    vec3 t;
    mat4 Q, T, TI, TIT, TIQTIT;
public:
    Hiperboloid(const vec3& _center, vec4 _r, vec3 _s, vec3 _t, Material* _material) {
        center = _center;
        r = _r;
        material = _material;
        s = _s;
        t = _t;
        Q = mat4(
            vec4(1 / (r.x * r.x), 0, 0, 0),
            vec4(0, 1 / (r.y * r.y), 0, 0),
            vec4(0, 0, -(1 / (r.z * r.z)), 0),
            vec4(0, 0, 0, -1)
        );
        T = mat4(vec4(s.x, 0, 0, 0),
            vec4(0, s.y, 0, 0),
            vec4(0, 0, s.z, 0),
            vec4(t.x, t.y, t.z, 1));
        TI = mat4(vec4(1 / s.x, 0, 0, 0),
            vec4(0, 1 / s.y, 0, 0),
            vec4(0, 0, 1 / s.z, 0),
            vec4(-(t.x / s.x), -(t.y / s.y), -(t.z / s.z), 1));
        TIT = mat4(
            vec4(1 / s.x, 0, 0, -(t.x / s.x)),
            vec4(0, 1 / s.y, 0, -(t.y / s.y)),
            vec4(0, 0, 1 / s.z, -(t.z / s.z)),
            vec4(0, 0, 0, 1)
        );
        TIQTIT = TI * Q * TIT;
    }



    vec3 gradf(vec4 r) {
        vec4 g = r * TIQTIT * 2;
        return vec3(g.x, g.y, g.z);
    }


    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TIQTIT), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float b = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TIQTIT), vec4(ray.start.x, ray.start.y, ray.start.z, 1)) + dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TIQTIT), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float c = dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TIQTIT), vec4(ray.start.x, ray.start.y, ray.start.z, 1));
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;

        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;

        hit.position = ray.start + ray.dir * hit.t;
        hit.material = material;
        if (hit.position.z > 0.2f || hit.position.z < -1) {
            hit.t = t1;
            hit.position = ray.start + ray.dir * hit.t;
            hit.material = material;
            if (hit.position.z > 0.2f || hit.position.z < -1) {
                Hit hit1;
                hit1.t = -1;
                return hit1;
            }
        }
        hit.normal = gradf((vec4(hit.position.x, hit.position.y, hit.position.z, 1))) / length(gradf((vec4(hit.position.x, hit.position.y, hit.position.z, 1))));
        return hit;
    }
};


class LightTube : public Intersectable {
    vec3 center;
    vec4 r;
    vec3 s;
    vec3 t;
    mat4 Q, T, TI, TIT, TIQTIT;
public:
    LightTube(const vec3& _center, vec4 _r, vec3 _s, vec3 _t, Material* _material) {
        center = _center;
        r = _r;
        material = _material;
        s = _s;
        t = _t;
        Q = mat4(
            vec4(1 / (r.x * r.x), 0, 0, 0),
            vec4(0, 1 / (r.y * r.y), 0, 0),
            vec4(0, 0, -(1 / (r.z * r.z)), 0),
            vec4(0, 0, 0, -1));
        T = mat4(vec4(s.x, 0, 0, 0),
            vec4(0, s.y, 0, 0),
            vec4(0, 0, s.z, 0),
            vec4(t.x, t.y, t.z, 1));
        TI = mat4(vec4(1 / s.x, 0, 0, 0),
            vec4(0, 1 / s.y, 0, 0),
            vec4(0, 0, 1 / s.z, 0),
            vec4(-(t.x / s.x), -(t.y / s.y), -(t.z / s.z), 1));
        TIT = mat4(
            vec4(1 / s.x, 0, 0, -(t.x / s.x)),
            vec4(0, 1 / s.y, 0, -(t.y / s.y)),
            vec4(0, 0, 1 / s.z, -(t.z / s.z)),
            vec4(0, 0, 0, 1));
        TIQTIT = TI * Q * TIT;
    }


    vec3 gradf(vec4 r) {
        vec4 g = r * TIQTIT * 2;
        return vec3(g.x, g.y, g.z);
    }


    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TIQTIT), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float b = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TIQTIT), vec4(ray.start.x, ray.start.y, ray.start.z, 1)) + dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TIQTIT), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float c = dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TIQTIT), vec4(ray.start.x, ray.start.y, ray.start.z, 1));
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;

        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;

        hit.position = ray.start + ray.dir * hit.t;
        hit.material = material;
        if (hit.position.z > 2.0f || hit.position.z < 1) {
            hit.t = t1;
            hit.position = ray.start + ray.dir * hit.t;
            hit.material = material;
            if (hit.position.z > 2.0f || hit.position.z < 1) {
                Hit hit1;
                hit1.t = -1;
                return hit1;
            }
        }
        hit.normal = gradf((vec4(hit.position.x, hit.position.y, hit.position.z, 1))) / length(gradf((vec4(hit.position.x, hit.position.y, hit.position.z, 1))));
        return hit;
    }
};



class Paraboloid : public Intersectable {
    vec3 center;
    vec4 r;
    vec3 s;
    vec3 t;
    mat4 Q, T, TI, TIT, TIQTIT;
public:
    Paraboloid(const vec3& _center, vec4 _r, vec3 _s, vec3 _t, Material* _material) {
        center = _center;
        r = _r;
        material = _material;
        s = _s;
        t = _t;
        Q = mat4(
            vec4(1 / (r.x * r.x), 0, 0, 0),
            vec4(0, 1 / (r.y * r.y), 0, 0),
            vec4(0, 0, 0, 1),
            vec4(0, 0, 0, 0));
        T = mat4(vec4(s.x, 0, 0, 0),
            vec4(0, s.y, 0, 0),
            vec4(0, 0, s.z, 0),
            vec4(t.x, t.y, t.z, 1));
        TI = mat4(vec4(1 / s.x, 0, 0, 0),
            vec4(0, 1 / s.y, 0, 0),
            vec4(0, 0, 1 / s.z, 0),
            vec4(-(t.x / s.x), -(t.y / s.y), -(t.z / s.z), 1));
        TIT = mat4(
            vec4(1 / s.x, 0, 0, -(t.x / s.x)),
            vec4(0, 1 / s.y, 0, -(t.y / s.y)),
            vec4(0, 0, 1 / s.z, -(t.z / s.z)),
            vec4(0, 0, 0, 1));
        TIQTIT = TI * Q * TIT;
    }


    vec3 gradf(vec4 r) {
        vec4 g = r * TIQTIT * 2;
        return vec3(g.x, g.y, g.z);
    }


    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TIQTIT), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float b = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TIQTIT), vec4(ray.start.x, ray.start.y, ray.start.z, 1)) + dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TIQTIT), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float c = dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TIQTIT), vec4(ray.start.x, ray.start.y, ray.start.z, 1));
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;

        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;

        hit.position = ray.start + ray.dir * hit.t;
        hit.material = material;
        if (hit.position.z < -1) {
            hit.t = t1;
            hit.position = ray.start + ray.dir * hit.t;
            hit.material = material;
            if (hit.position.z < -1) {
                Hit hit1;
                hit1.t = -1;
                return hit1;
            }
        }
        hit.normal = gradf((vec4(hit.position.x, hit.position.y, hit.position.z, 1))) / length(gradf((vec4(hit.position.x, hit.position.y, hit.position.z, 1))));
        return hit;
    }
};

class Room : public Intersectable {
    vec3 center, s, t;
    vec4 r;
    mat4 Q, T, TI, TIT, TIQTIT;
public:
    Room(const vec3& _center, vec4 _r, vec3 _s, vec3 _t, Material* _material) {
        center = _center;
        r = _r;
        material = _material;
        s = _s;
        t = _t;
        Q = mat4(
            vec4(1 / (r.x * r.x), 0, 0, 0),
            vec4(0, 1 / (r.y * r.y), 0, 0),
            vec4(0, 0, 1 / (r.z * r.z), 0),
            vec4(0, 0, 0, -1));
        T = mat4(vec4(s.x, 0, 0, 0),
            vec4(0, s.y, 0, 0),
            vec4(0, 0, s.z, 0),
            vec4(t.x, t.y, t.z, 1));
        TI = mat4(vec4(1 / s.x, 0, 0, 0),
            vec4(0, 1 / s.y, 0, 0),
            vec4(0, 0, 1 / s.z, 0),
            vec4(-(t.x / s.x), -(t.y / s.y), -(t.z / s.z), 1));
        TIT = mat4(
            vec4(1 / s.x, 0, 0, -(t.x / s.x)),
            vec4(0, 1 / s.y, 0, -(t.y / s.y)),
            vec4(0, 0, 1 / s.z, -(t.z / s.z)),
            vec4(0, 0, 0, 1));
        TIQTIT = TI * Q * TIT;
    }

    vec3 gradf(vec4 r) {
        vec4 g = r * TIQTIT * 2;
        return vec3(g.x, g.y, g.z);
    }


    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TIQTIT), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float b = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TIQTIT), vec4(ray.start.x, ray.start.y, ray.start.z, 1)) + dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TIQTIT), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float c = dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TIQTIT), vec4(ray.start.x, ray.start.y, ray.start.z, 1));
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;

        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;

        hit.position = ray.start + ray.dir * hit.t;
        hit.material = material;
        if (hit.position.z > 0.97f) {
            hit.t = t1;
            hit.position = ray.start + ray.dir * hit.t;
            hit.material = material;
            if (hit.position.z > 0.97f) {
                Hit hit1;
                hit1.t = -1;
                return hit1;
            }
        }
        hit.normal = gradf((vec4(hit.position.x, hit.position.y, hit.position.z, 1))) / length(gradf((vec4(hit.position.x, hit.position.y, hit.position.z, 1))));
        return hit;
    }
};


class Ellipsoid : public Intersectable {
    vec3 center, s, t;
    vec4 r;
    mat4 Q, T, TI, TIT, TIQTIT;
public:
    Ellipsoid(const vec3& _center, vec4 _r, vec3 _s, vec3 _t, Material* _material) {
        center = _center;
        r = _r;
        material = _material;
        s = _s;
        t = _t;
        Q = mat4(
            vec4(1 / (r.x * r.x), 0, 0, 0),
            vec4(0, 1 / (r.y * r.y), 0, 0),
            vec4(0, 0, 1 / (r.z * r.z), 0),
            vec4(0, 0, 0, -1));
        T = mat4(vec4(s.x, 0, 0, 0),
            vec4(0, s.y, 0, 0),
            vec4(0, 0, s.z, 0),
            vec4(t.x, t.y, t.z, 1));
        TI = mat4(vec4(1 / s.x, 0, 0, 0),
            vec4(0, 1 / s.y, 0, 0),
            vec4(0, 0, 1 / s.z, 0),
            vec4(-(t.x / s.x), -(t.y / s.y), -(t.z / s.z), 1));
        TIT = mat4(
            vec4(1 / s.x, 0, 0, -(t.x / s.x)),
            vec4(0, 1 / s.y, 0, -(t.y / s.y)),
            vec4(0, 0, 1 / s.z, -(t.z / s.z)),
            vec4(0, 0, 0, 1));
        TIQTIT = TI * Q * TIT;
    }

    vec3 gradf(vec4 r) {
        vec4 g = r * TIQTIT * 2;
        return vec3(g.x, g.y, g.z);
    }


    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TIQTIT), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float b = dot(vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0) * (TIQTIT), vec4(ray.start.x, ray.start.y, ray.start.z, 1)) + dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TIQTIT), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
        float c = dot(vec4(ray.start.x, ray.start.y, ray.start.z, 1) * (TIQTIT), vec4(ray.start.x, ray.start.y, ray.start.z, 1));
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;

        if (t1 <= 0) return hit;

        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = gradf((vec4(hit.position.x, hit.position.y, hit.position.z, 1))) / length(gradf((vec4(hit.position.x, hit.position.y, hit.position.z, 1))));
        hit.material = material;

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

struct LightPoint {
    vec3 pos;
    LightPoint(vec3 _pos) {
        pos = normalize(_pos);
    }
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
    std::vector<Intersectable*> objects;
    std::vector<Light*> lights;
    std::vector<LightPoint*> lightPoints;
    Camera camera;
    vec3 La;
    vec3 sky = vec3(5.0f, 5.0f, 10.0f);
    vec3 sun = vec3(15.0f, 15.0f, 15.0f);
public:
    void build() {
        vec3 eye = vec3(0.0f, 1.9f, 0.0f), vup = vec3(0.0f, 0.0f, 1.0f), lookat = vec3(0.0f, 0.0f, 0.2f);

        float fov = 90 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.15f, 0.15f, 0.15f);

        vec3 lightDirection(3.0f, 3.0f, 3.0f), Le(2.0f, 2.0f, 2.0f);
        lights.push_back(new Light(lightDirection, Le));

        vec3 goldN(0.17f, 0.35f, 1.5f);
        vec3 goldKappa(3.1f, 2.7f, 1.9f);
        vec3 silverN(0.14f, 0.16f, 0.13f);
        vec3 silverKappa(4.1f, 2.3f, 3.1f);
        Material* goldMaterial = new ReflectiveMaterial(goldN, goldKappa);
        Material* silverMaterial = new ReflectiveMaterial(silverN, silverKappa);

        vec3 ks(0.5f, 0.5f, 0.5f);
        vec3 lightOrange(0.8f, 0.25f, 0.25f);
        vec3 lightBlue(0.0f, 0.25f, 0.25f);
        vec3 roomColor(0.35f, 0.35f, 0.5f);
        Material* roomRough = new RoughMaterial(roomColor, ks, 200);
        Material* lightOrangeRough = new RoughMaterial(lightOrange, ks, 50);
        Material* lightBlueRough = new RoughMaterial(lightBlue, ks, 50);


        vec3 origo(0.0f, 0.0f, 0.97f);
        while (lightPoints.size() < 150) {
            float randX = -0.48 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (0.48 - (-0.48))));
            float randY = -0.48 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (0.48 - (-0.48))));

            float distance = sqrt((origo.x - randX) * (origo.x - randX) + (origo.y - randY) * (origo.y - randY));

            if (distance < 0.48f) {
                lightPoints.push_back(new LightPoint(vec3(randX, randY, 0.97f)));
            }
        }

        objects.push_back(new Room(vec3(0.0f, 0.0f, 0.0f), vec4(2.0f, 2.0f, 1.0f), vec3(1.0f, 1.0f, 1.0f), vec3(0.0f, 0.0f, 0.0f), roomRough));
        objects.push_back(new LightTube(vec3(0.0f, 0.0f, 0.0f), vec4(1.3f, 1.3f, 1), vec3(0.15f, 0.15f, 0.15f), vec3(0, 0, -0.25f), silverMaterial));

        objects.push_back(new Ellipsoid(vec3(0.0f, 0.0f, 0.0f), vec4(2.7f, 0.7f, 0.8f), vec3(0.3f, 0.3f, 0.3f), vec3(-0.4f, -0.6f, -0.7f), lightBlueRough));
        objects.push_back(new Hiperboloid(vec3(0.0f, 0.0f, 0.0f), vec4(0.3f, 0.3f, 0.3f), vec3(0.32f, 0.32f, 0.32f), vec3(-0.6f, -1.5f, -0.3f), lightOrangeRough));
        objects.push_back(new Paraboloid(vec3(0.0f, 0.0f, 0.0f), vec4(1.6f, 1.6f, 1), vec3(0.08f, 0.08f, 0.08f), vec3(1, -0.1f, 0), goldMaterial));
        
        objects.push_back(new Ellipsoid(vec3(0, 0, 0), vec4(0.7f, 0.7f, 0.7f), vec3(0.1f, 0.1f, 0.1f), vec3(-1.2f, 0.2f, -0.7f), lightOrangeRough));
        objects.push_back(new Ellipsoid(vec3(0, 0, 0), vec4(0.7f, 0.7f, 0.7f), vec3(0.1f, 0.1f, 0.1f), vec3(-1.1f, 0.3f, -0.75f), lightOrangeRough));
        objects.push_back(new Ellipsoid(vec3(0, 0, 0), vec4(0.7f, 0.7f, 0.7f), vec3(0.1f, 0.1f, 0.1f), vec3(-1.0f, 0.2f, -0.8f), lightBlueRough));
        objects.push_back(new Ellipsoid(vec3(0, 0, 0), vec4(0.7f, 0.7f, 0.7f), vec3(0.1f, 0.1f, 0.1f), vec3(-0.5f, 0.4f, -0.9f), lightOrangeRough));

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
            Hit hit = object->intersect(ray);
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray, float max = -1) {
        for (Intersectable* object : objects) {
            float t = object->intersect(ray).t;
            if (t > 0 && (max < 0 || t > max)) return true;
        }
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        if (depth > 5) return La;
        Hit hit = firstIntersect(ray);
        vec3 sunDir = normalize(lights.at(0)->direction);
        if (hit.t < 0) {
            return (sky + sun * pow(dot(ray.dir, sunDir), 10));
        }

        vec3 outRadiance(0.0f, 0.0f, 0.0f);

        if (hit.material->type == ROUGH) {
            outRadiance = hit.material->ka * La;
            for (LightPoint* lightPoint : lightPoints) {
                vec3 rayDir = lightPoint->pos - hit.position;
                Ray pixelToLight(hit.position + hit.normal * epsilon, normalize(rayDir));
                float cosTheta = dot(hit.normal, normalize(rayDir));

                if (cosTheta > 0 && !shadowIntersect(pixelToLight, length(rayDir))) {
                    vec3 lightDir = hit.position - lightPoint->pos;


                    float dotProd = dot(-normalize(rayDir), vec3(0.0f, 0.0f, -1.0f));

                    float omega = (0.48f * 0.48f * M_PI) / lightPoints.size() * dotProd / (dot((lightPoint->pos - hit.position), (lightPoint->pos - hit.position)));
                    outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, normalize(rayDir)), depth + 1) * hit.material->kd * cosTheta * omega;

                    vec3 halfway = normalize(-ray.dir + lightPoint->pos);
                    float cosDelta = dot(hit.normal, normalize(halfway));

                    if (cosDelta > 0) outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, normalize(rayDir)), depth + 1) * hit.material->ks * powf(cosDelta, hit.material->shininess) * omega;
                }

            }

        }

        if (hit.material->type == REFLECTIVE) {
            vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
            float cosa = -dot(ray.dir, hit.normal);
            vec3 one(1.0f, 1.0f, 1.0f);
            vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
            outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
        }

        return outRadiance;
    }
};

GPUProgram gpuProgram;
Scene scene;

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
    unsigned int vao = 0, textureId = 0;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight)
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        unsigned int vbo;
        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

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
        glBindVertexArray(vao);
        int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
        const unsigned int textureUnit = 0;
        if (location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
    std::vector<vec4> image(windowWidth * windowHeight);

    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("\n\n\n\n Rendering time: %d milliseconds\n\n\n\n", (timeEnd - timeStart));

    fullScreenTexturedQuad->LoadTexture(image);

    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
    glutPostRedisplay();
}
