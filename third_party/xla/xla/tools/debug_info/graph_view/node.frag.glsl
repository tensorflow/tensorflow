#version 300 es
precision highp float;
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

  in vec4 vColor;
  in vec2 vUV;
  in float vSelectionState;
  in float vBodyBoundary;
  out vec4 fragColor;

  void main() {
    // Discard pixels outside the circle
    vec2 uv = vUV - 0.5;
    float rSq = dot(uv, uv);
    if (rSq > 0.25) {
      discard;
    }

    // Ensure negative boundaries don't become positive when squared
    float clampedBoundary = max(vBodyBoundary, 0.0);
    float innerRSq = clampedBoundary * clampedBoundary;

    if (rSq > innerRSq) {
      if (vSelectionState > 0.5 && vSelectionState < 1.5) {
        fragColor = vec4(0.2, 0.2, 0.2, 1.0); // Selected -> Dark
      } else if (vSelectionState > 1.5) {
        fragColor = vec4(0.6, 0.6, 0.6, 1.0); // Neighbor -> Light grey
      } else {
        fragColor = vColor; // Normal -> No border
      }
    } else {
      fragColor = vColor;
    }
  }
