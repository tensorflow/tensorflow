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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_TYPES_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_TYPES_H_

#include <string>
#include <vector>

#include "absl/types/variant.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace gpu {

// GpuVersion is used to abstract Gpu hardware version. On Cuda platform,
// it comprises a pair of integers denoting major and minor version.
// On ROCm platform, it comprises one integer for AMD GCN ISA version.
using GpuVersion = absl::variant<std::pair<int, int>, int>;

// A struct to carry around compiled results by the GPU assembler.
struct GpuTargetBinary {
  GpuTargetBinary(const GpuTargetBinary& other) = delete;
  GpuTargetBinary(GpuTargetBinary&& other) = default;

  // The text format of the compiled result, e.g. PTX.
  std::string text;

  // The actual compiled binary.
  std::vector<tensorflow::uint8> binary;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_TYPES_H_
