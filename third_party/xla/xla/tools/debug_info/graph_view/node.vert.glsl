#version 300 es
// Copyright 2026 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

in vec2 aPosition; // Quad vertex offset [-0.5, 0.5]
in vec2 aCenter;   // Instance position
in float aSelectionState; // 0=normal, 1=selected, 2=neighbor
in float aDiffScore;

uniform mat3 uMatrix;
uniform float uNodeSize;
uniform float uZoom;
uniform float uStrokeThicknessPixels;

out vec4 vColor;
out vec2 vUV;
out float vSelectionState;
out float vBodyBoundary;

// The diff score thresholds and hex color here are mirrored from the
// Comparison Tool's C++ thresholds:
//   <0.0   => Grey (#808080)
//   0.0    => Green (#99ff99)
//   1.0    => Light Yellow-Green (#c0f580)
//   5.0    => Yellow-Green (#e0ee40)
//   10.0   => Yellow (#eeee00)
//   30.0   => Yellow-Orange (#ffc000)
//   60.0   => Orange (#ff8000)
//   >=100.0 => Red (#ff1717)
vec4 getColorForScore(float score) {
    if (score < 0.0) return vec4(0.82, 0.82, 0.82, 1.0);
    if (score >= 100.0) return vec4(1.0, 0.090, 0.090, 1.0);
    if (score < 1.0) {
        return mix(vec4(0.6, 1.0, 0.6, 1.0), vec4(0.753, 0.961, 0.502, 1.0), score);
    } else if (score < 5.0) {
        return mix(vec4(0.753, 0.961, 0.502, 1.0), vec4(0.878, 0.933, 0.251, 1.0), (score - 1.0) / 4.0);
    } else if (score < 10.0) {
        return mix(vec4(0.878, 0.933, 0.251, 1.0), vec4(0.933, 0.933, 0.0, 1.0), (score - 5.0) / 5.0);
    } else if (score < 30.0) {
        return mix(vec4(0.933, 0.933, 0.0, 1.0), vec4(1.0, 0.753, 0.0, 1.0), (score - 10.0) / 20.0);
    } else if (score < 60.0) {
        return mix(vec4(1.0, 0.753, 0.0, 1.0), vec4(1.0, 0.502, 0.0, 1.0), (score - 30.0) / 30.0);
    } else {
        return mix(vec4(1.0, 0.502, 0.0, 1.0), vec4(1.0, 0.090, 0.090, 1.0), (score - 60.0) / 40.0);
    }
}

void main() {
    vColor = getColorForScore(aDiffScore);
    vUV = aPosition + 0.5; // [0, 1]
    vSelectionState = aSelectionState;

    float strokeWorld = uStrokeThicknessPixels / uZoom;
    float bodySize = uNodeSize;

    // Focus on drawing nodes with high diff score at small scale
    if (uZoom < 0.1 && aDiffScore > 0.0) {
      float zoomFactor = clamp((0.1 - uZoom) / 0.1, 0.0, 1.0);
      float diffFactor = clamp(aDiffScore / 20.0, 0.0, 5.0);
      bodySize *= (1.0 + zoomFactor * diffFactor);
    }

    if (aSelectionState > 0.5 && aSelectionState < 1.5) bodySize *= 1.5; // Selected

    float quadSize = bodySize;
    if (aSelectionState > 1.5) { // Neighbor
      quadSize = bodySize + 2.0 * strokeWorld;
    }

    vBodyBoundary = 0.5 - (strokeWorld / quadSize);

    // Apply quad offset in world Space
    vec2 worldPos = aCenter + aPosition * quadSize;
    vec3 pos = uMatrix * vec3(worldPos, 1.0);

    // Depth based on diff score (higher score -> smaller Z/closer)
    float depth = -clamp((aDiffScore + 1.0) / 101.0, 0.0, 1.0);
    gl_Position = vec4(pos.xy, depth, 1.0);
  }
