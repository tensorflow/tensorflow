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
    if (vSelectionState > 0.5 && vSelectionState < 1.5) {
      fragColor = vec4(0.2, 0.2, 0.2, 1.0); // Selected -> Dark
    } else if (vSelectionState > 1.5) {
      fragColor = vec4(0.6, 0.6, 0.6, 1.0); // Neighbor -> Light grey
    } else {
      fragColor = vColor;
    }
  }
