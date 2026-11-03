/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_GPU_TRANSPOSE_FOLDING_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_GPU_TRANSPOSE_FOLDING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Folds transposes into dot and convolution dimension numbers.
//
// For dots, folds chains of transposes and bitcasts feeding dot operands into
// the dimension numbers whenever that leaves the dot result unchanged (see
// DotOperandDims::FoldBackward; layout-changing bitcasts count as the logical
// transposes they represent) and the source's layout is supported by GEMMs
// (including the minor contracting dimensions requirement, see
// DotRequiresMinorContractingDims).
//
// For convolutions, folds transpose operands into the input and kernel
// dimension numbers if `fold_into_convolutions` is set.
class GpuTransposeFolding : public HloModulePass {
 public:
  GpuTransposeFolding(const se::GpuComputeCapability& gpu_version,
                      bool fold_into_convolutions)
      : gpu_version_(gpu_version),
        fold_into_convolutions_(fold_into_convolutions) {}

  absl::string_view name() const override { return "gpu-transpose-folding"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
  bool fold_into_convolutions_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_GPU_TRANSPOSE_FOLDING_H_
