/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_OPTIMIZE_INPUT_OUTPUT_BUFFER_ALIAS_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_OPTIMIZE_INPUT_OUTPUT_BUFFER_ALIAS_H_

#include <cstdint>
#include <functional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

// This pass finds input and output buffers that can be aliased, and writes the
// alias config into the HloModule.
//
// The input and the output buffers can be in any shape, and each output buffer
// can alias with an input buffer with the same shape. Each input buffer may
// only alias with a single output buffer. For example, for the following
// parameter and the output buffers,
//
//  Parameters : { P1(f32[3]), P2(s32[3]), P3(f32[3,12]), P4(f32[16,12]), ... }
//  Outputs    : { O1(s32[3]), O2(f32[3]), O3(f32[16,12]), ... }
//
// one potential aliasing would be (O1, P2), (O2, P1), (O3, P4), ..
class OptimizeInputOutputBufferAlias : public HloModulePass {
 public:
  OptimizeInputOutputBufferAlias() = default;
  explicit OptimizeInputOutputBufferAlias(
      bool registered_buffer_donor_only,
      std::function<int64_t(const Shape&)> shape_size_fn =
          [](const Shape& shape) { return ShapeUtil::ByteSizeOf(shape); })
      : registered_buffer_donor_only_(registered_buffer_donor_only),
        shape_size_fn_(shape_size_fn) {}
  ~OptimizeInputOutputBufferAlias() override = default;

  absl::string_view name() const override {
    return "optimize_input_output_buffer_alias";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  friend class OptimizeInputOutputBufferAliasTest;

  // If true, we only consider the registered buffer donor in
  // HloBufferDonorConfig, ignoring unregistered input parameters. If false, we
  // treat all input parameters as buffer donors.
  bool registered_buffer_donor_only_ = false;

  // Match buffer donors and donees and save the matched paired in the
  // alias_config. The availability of buffer donors is controlled by the flag
  // registered_buffer_donor_only_.
  absl::StatusOr<bool> Build(absl::Span<const Shape> input_shapes,
                             const Shape& output_shape,
                             HloInputOutputAliasConfig* alias_config,
                             HloBufferDonorConfig* buffer_donor_config);

  std::function<int64_t(const Shape&)> shape_size_fn_ = [](const Shape& shape) {
    return ShapeUtil::ByteSizeOf(shape);
  };
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_OPTIMIZE_INPUT_OUTPUT_BUFFER_ALIAS_H_
