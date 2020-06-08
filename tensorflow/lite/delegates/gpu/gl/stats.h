/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_STATS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_STATS_H_

#include <numeric>

#include "absl/strings/str_cat.h"

namespace tflite {
namespace gpu {
namespace gl {

// A collection of compile-time stats exposed via API.
struct CompilerStats {};

struct ObjectStats {
  // Number of allocated objects.
  int32_t count = 0;

  // Total bytes allocated.
  int64_t total_bytes = 0;
};

struct ObjectsStats {
  ObjectStats buffers;

  ObjectStats textures;
};

// A collection of runtime-time stats exposed via API.
struct RuntimeStats {
  ObjectsStats internal_objects;

  ObjectsStats const_objects;

  ObjectsStats external_objects;
};

inline std::string ToString(const ObjectStats& stats) {
  return absl::StrCat("count = ", stats.count,
                      ", total bytes = ", stats.total_bytes);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_STATS_H_
