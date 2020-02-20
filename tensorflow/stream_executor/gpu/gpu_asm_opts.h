/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_ASM_OPTS_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_ASM_OPTS_H_

#include <string>
#include <tuple>

#include "absl/strings/string_view.h"

namespace stream_executor {
// Compilation options for compiling ptxas.
struct GpuAsmOpts {
  // Disable Cuda ptxas optimizations.
  bool disable_gpuasm_optimizations;

  // Cuda directory which would be searched first.
  std::string preferred_cuda_dir;

  explicit GpuAsmOpts(bool disable_gpuasm_optimizations = false,
                      absl::string_view preferred_cuda_dir = "")
      : disable_gpuasm_optimizations(disable_gpuasm_optimizations),
        preferred_cuda_dir(preferred_cuda_dir) {}

  using PtxOptionsTuple = std::tuple<bool, std::string>;

  PtxOptionsTuple ToTuple() {
    return std::make_tuple(disable_gpuasm_optimizations, preferred_cuda_dir);
  }
};
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_ASM_OPTS_H_
