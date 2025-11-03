#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba8) readonly  uniform image2D inImg;
layout(binding = 1, rgba8) writeonly uniform image2D outImg;

// --- Tunables ---
const int  R = 6;              // circle radius for FAST-9
const int  N = 9;              // contiguous arc length
const float THRESH = 0.3;     // intensity threshold in [0,1]

// Circle offsets for radius 3, starting at top and going clockwise
const ivec2 circle[16] = ivec2[16](
    ivec2( 0,-6), ivec2( 2,-6), ivec2( 4,-5), ivec2( 5,-4),
    ivec2( 6, 0), ivec2( 5, 4), ivec2( 4, 5), ivec2( 2, 6),
    ivec2( 0, 6), ivec2(-2, 6), ivec2(-4, 5), ivec2(-5, 4),
    ivec2(-6, 0), ivec2(-5,-4), ivec2(-4,-5), ivec2(-2,-6)
);

float luminance(vec4 rgba) {
    return dot(rgba.rgb, vec3(0.299, 0.587, 0.114));
}

bool isCorner(ivec2 p, ivec2 size) {
    if (p.x < R || p.y < R || p.x >= size.x - R || p.y >= size.y - R)
        return false;

    float I0 = luminance(imageLoad(inImg, p));
    bool bright[32];
    bool dark[32];

    for (int i = 0; i < 16; ++i) {
        float Ii = luminance(imageLoad(inImg, p + circle[i]));
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

void drawCircle(ivec2 center, ivec2 size) {
    // Red overlay color
    vec4 red = vec4(1.0, 0.0, 0.0, 1.0);
    int R2 = R * R;

    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            ivec2 q = center + ivec2(dx, dy);
            if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) continue;

            int d2 = dx*dx + dy*dy;
            if (abs(d2 - R2) <= 2) { // pixels close to radius
                vec4 orig = imageLoad(inImg, q);
                vec4 blended = mix(orig, red, 0.8); // blend red with original
                imageStore(outImg, q, blended);
            }
        }
    }
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(inImg);

    if (p.x >= size.x || p.y >= size.y) return;

    bool corner = isCorner(p, size);

    if (corner) {
        drawCircle(p, size);
    } else {
        // If not a corner, just copy the input pixel
        vec4 px = imageLoad(inImg, p);
        imageStore(outImg, p, px);
    }
}
