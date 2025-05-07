/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_CUSTOM_NVPTX_COMPILER_H_
#define XLA_SERVICE_GPU_CUSTOM_NVPTX_COMPILER_H_

#include "xla/service/gpu/nvptx_compiler.h"

namespace xla {
namespace gpu {

class CustomNVPTXCompiler : public NVPTXCompiler {
 public:
  explicit CustomNVPTXCompiler();

  absl::Status RunFusionPasses(
      HloModule* hlo_module, const Compiler::TargetConfig& gpu_target_config,
      tsl::thread::ThreadPool* thread_pool,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn) final;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUSTOM_NVPTX_COMPILER_H_
