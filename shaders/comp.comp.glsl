#version 450
layout(local_size_x = 16, local_size_y = 16) in;

// Packed RGBA8 per pixel
layout(binding = 0, std430) readonly buffer InputBuffer {
    uint data[];
} inBuf;
layout(binding = 1, std430) writeonly buffer OutputBuffer {
    uint data[];
} outBuf;

//Corner-liste
layout(binding = 2, std430) buffer CornerList {
    uint capacity;   // max. Anzahl Elemente in coords[]
    uint count;      // wird per atomicAdd inkrementiert
    uvec2 coords[];  // Laufzeit-Array der Corner-Koordinaten
} corners;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
} pc;

const int  R = 3;
const int  N = 9;
const float THRESH = 0.3;

const ivec2 circle[16] = ivec2[16](
    ivec2( 0,-3), ivec2( 1,-3), ivec2( 2,-2), ivec2( 3,-1),
    ivec2( 3, 0), ivec2( 3, 1), ivec2( 2, 2), ivec2( 1, 3),
    ivec2( 0, 3), ivec2(-1, 3), ivec2(-2, 2), ivec2(-3, 1),
    ivec2(-3, 0), ivec2(-3,-1), ivec2(-2,-2), ivec2(-1,-3)
);

// index helpers
uint idx(ivec2 p) { return uint(p.y) * pc.width + uint(p.x); }

// load/store with pack/unpack (assumes input is RGBA8 in [0,255])
vec4 loadRGBA(ivec2 p) {
    if (p.x < 0 || p.y < 0 || p.x >= int(pc.width) || p.y >= int(pc.height)) return vec4(0.0);
    return unpackUnorm4x8(inBuf.data[idx(p)]); // returns vec4 in [0,1]
}
void storeRGBA(ivec2 p, vec4 c) {
    if (p.x < 0 || p.y < 0 || p.x >= int(pc.width) || p.y >= int(pc.height)) return;
    outBuf.data[idx(p)] = packUnorm4x8(clamp(c, 0.0, 1.0));
}

ivec2 imageSize() { return ivec2(int(pc.width), int(pc.height)); }
float luminance(vec4 rgba) { return dot(rgba.rgb, vec3(0.299, 0.587, 0.114)); }

bool isCorner(ivec2 p, ivec2 size) {
    if (p.x < R || p.y < R || p.x >= size.x - R || p.y >= size.y - R) return false;
    float I0 = luminance(loadRGBA(p));
    bool bright[32]; bool dark[32];
    for (int i = 0; i < 16; ++i) {
        float Ii = luminance(loadRGBA(p + circle[i]));
        bright[i] = (Ii >= I0 + THRESH);
        dark[i]   = (Ii <= I0 - THRESH);
        bright[i+16] = bright[i];
        dark[i+16]   = dark[i];
    }
    int runB = 0, runD = 0;
    for (int i = 0; i < 16 + N - 1; ++i) {
        runB = bright[i] ? (runB + 1) : 0;
        runD = dark[i]   ? (runD + 1) : 0;
        if (runB >= N || runD >= N) return true;
    }
    return false;
}

// RACE-FREE Variante: Jeder Thread schreibt NUR sein eigenes Pixel.
bool liesOnCircleOfCorner(ivec2 p, ivec2 size) {
    int R2 = R*R;
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            int d2 = dx*dx + dy*dy;
            if (abs(d2 - R2) > 2) continue;
            ivec2 c = p - ivec2(dx, dy);
            if (c.x < 0 || c.y < 0 || c.x >= size.x || c.y >= size.y) continue;
            if (isCorner(c, size)) return true;
        }
    }
    return false;
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize();
    if (p.x >= size.x || p.y >= size.y) return;

    vec4 color = loadRGBA(p);

    // Markieren: roter Punkt im Zentrum ODER roter Ring um Corner, ohne Nachbarpixel aktiv zu überschreiben
    if (isCorner(p, size)) {
        color = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (liesOnCircleOfCorner(p, size)) {
        color = mix(color, vec4(1.0, 0.0, 0.0, 1.0), 0.8);
    }

    storeRGBA(p, color);
    
    if (isCorner(p, size)) {
        uint i = atomicAdd(corners.count, 1u);
        if (i < corners.capacity) {
            corners.coords[i] = uvec2(p);
        }
    }
}