/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/ptx_compiler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {
absl::StatusOr<std::vector<uint8_t>> CompileGpuAsmUsingLibNvPtxCompiler(
    const CudaComputeCapability& cc, const std::string& ptx, GpuAsmOpts options,
    bool cancel_if_reg_spill) {
  return absl::UnimplementedError(
      "XLA was built without libnvptxcompiler support.");
}

absl::StatusOr<SemanticVersion> GetLibNvPtxCompilerVersion() {
  return absl::UnimplementedError(
      "XLA was built without libnvptxcompiler support.");
}
}  // namespace stream_executor
