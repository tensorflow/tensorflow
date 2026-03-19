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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CONVERT_TRITON_GEMM_CONFIG_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CONVERT_TRITON_GEMM_CONFIG_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Annotates instructions inside the triton_gemm fusions with the tiling
// parameters from its backend config.
//
// Replaces the fusion kind with "__triton_nested_gemm_fusion" and sets the
// fusion's backend config a BlockLevelFusionConfig, derived from
// TritonGemmConfig.
class ConvertTritonGemmConfig : public HloModulePass {
 public:
  explicit ConvertTritonGemmConfig(
      const se::DeviceDescription& device_description,
      mlir::MLIRContext* mlir_context)
      : device_description_(device_description), mlir_context_(mlir_context) {}

  absl::string_view name() const override {
    return "convert_triton_gemm_config";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::DeviceDescription device_description_;
  mlir::MLIRContext* mlir_context_;
};

// Returns block level parameters based on tile sizes for the root of the
// analysis that satisfy the requirements of the `dot`. That is, the tile sizes
// need to satisfy the constraints of the analysis and map to the given `config`
// of the dot.
absl::StatusOr<BlockLevelParameters> FindBlockLevelParameters(
    HloInstruction* dot, const TritonGemmConfig& config,
    mlir::MLIRContext* mlir_context,
    const se::DeviceDescription& device_description);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CONVERT_TRITON_GEMM_CONFIG_H_
