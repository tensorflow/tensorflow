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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_ALIAS_IN_PLACE_OUTPUTS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_ALIAS_IN_PLACE_OUTPUTS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace mlir {
class MLIRContext;
}  // namespace mlir

namespace xla::gpu {

// Annotates GPU operations whose output buffer can overwrite an operand buffer
// in place with `output_to_operand_aliasing`, eliminating copies.
//
// Supported operations:
//   * Triton fusions (GEMMs, elementwise): an operand is aliased when it
//     reaches the fusion root through a single data-flow path consisting of
//     pure elementwise ops or bitcasts that cancel, and its shape is
//     compatible with the root.
//   * cuBLASLt custom calls (`__cublas$lt$matmul`, `__cublas$lt$matmul$f8`,
//     `__cublas$lt$matmul$mx`, `__cublas$lt$groupedMatmul`): the bias/C
//     operand is aliased when `beta != 0` and its shape matches the output.
//
// In both cases, the aliased operand must be a writable intermediate whose
// only other users (if any) precede this operation in the dataflow graph.
//
// Must run before copy insertion.
class AliasInPlaceOutputs : public HloModulePass {
 public:
  explicit AliasInPlaceOutputs(mlir::MLIRContext* mlir_context)
      : mlir_context_(mlir_context) {}

  absl::string_view name() const override { return "alias_in_place_outputs"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  mlir::MLIRContext* mlir_context_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_ALIAS_IN_PLACE_OUTPUTS_H_
