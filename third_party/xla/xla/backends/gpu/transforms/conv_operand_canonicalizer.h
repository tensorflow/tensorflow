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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CONV_OPERAND_CANONICALIZER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CONV_OPERAND_CANONICALIZER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Canonicalizes convolution operands.
//
// 1. Transforms s32 constants -> s32 convert(s8 constant)
// If a s32 constant is equivalent to an s8 constant, i.e. the constant contains
// values in the range [-128, 127], convert the constant to s8 and add a convert
// op to bring the operand back to s32.
//
// 2. Removes redundant converts, e.g. s32 convert(s32 convert(s8)) -> s32
// convert(s8).
//
// 3. Transforms SpatialOp(s32 convert(s8)) -> s32 convert(SpatialOp(s8)).
// Commutes convert op over spatial operations (e.g. Reshape, Transpose,
// Broadcast, Pad, Slice) and moves the convert to the convolution operand.

class ConvOperandCanonicalizer : public HloModulePass {
 public:
  ConvOperandCanonicalizer() = default;

  absl::string_view name() const override {
    return "conv-operand-canonicalizer";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CONV_OPERAND_CANONICALIZER_H_
