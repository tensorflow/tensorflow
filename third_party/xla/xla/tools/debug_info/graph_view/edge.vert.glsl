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

in float aT;
in vec2 aEndpointU;
in vec2 aEndpointV;
in vec2 aRadii;

uniform mat3 uMatrix;

void main() {
  float dx = aEndpointV.x - aEndpointU.x;
  vec2 p0 = aEndpointU;
  vec2 p3 = aEndpointV;

  if (dx > 0.0) {
    p0.x += aRadii.x;
    p3.x -= aRadii.y;
  } else if (dx < 0.0) {
    p0.x -= aRadii.x;
    p3.x += aRadii.y;
  }

  float dxNew = p3.x - p0.x;
  vec2 p1 = vec2(p0.x + dxNew / 2.0, p0.y);
  vec2 p2 = vec2(p3.x - dxNew / 2.0, p3.y);

  float t = aT;
  float t2 = t * t;
  float t3 = t2 * t;
  float ut = 1.0 - t;
  float ut2 = ut * ut;
  float ut3 = ut2 * ut;

  vec2 pos = ut3 * p0 + 3.0 * ut2 * t * p1 + 3.0 * ut * t2 * p2 + t3 * p3;

  vec3 worldPos = uMatrix * vec3(pos, 1.0);
  // Edges are behind nodes
  gl_Position = vec4(worldPos.xy, 0.5, 1.0);
}
