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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_OPTIMIZE_INPUT_OUTPUT_BUFFER_ALIAS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_OPTIMIZE_INPUT_OUTPUT_BUFFER_ALIAS_H_

#include <cstdint>
#include <functional>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

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
    return "optimize_input_output_buffer_alias.h";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  friend class OptimizeInputOutputBufferAliasTest;

  // If true, we only consider the registered buffer donors, ignoring
  // unregistered input parameters. If false, we treat all input parameters as
  // buffer donors.
  bool registered_buffer_donor_only_ = false;

  // Match buffer donors and donees and save the matched pairs in the
  // alias_config. The range of available buffer donors is controlled by the
  // flag registered_buffer_donor_only_.
  StatusOr<bool> Build(absl::Span<const Shape> input_shapes,
                       const Shape& output_shape,
                       HloInputOutputAliasConfig* alias_config,
                       HloBufferDonorConfig* buffer_donor_config);

  // For each input-output alias, we add control dependencies to avoid the
  // alias-miss in the run-time. An example is listed below.
  //
  // Parameter P1 is used in 3 instructions (I1, I2, I3). I3 is an output and we
  // match P1 and I3 as the input-output alias. P1 can donate its memory only
  // after the 3 instructions are finished. If I3 seeks the donated memory
  // before I1 and I2 finish, there will be a alias-miss in the runtime. We add
  // control dependencies I1-I3, I2-I3 such that I3 is the last instruction
  // among these 3 to avoid the alias-miss.
  StatusOr<bool> AddControlDependencyForAlias(HloModule* module);

  std::function<int64_t(const Shape&)> shape_size_fn_ = [](const Shape& shape) {
    return ShapeUtil::ByteSizeOf(shape);
  };
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_OPTIMIZE_INPUT_OUTPUT_BUFFER_ALIAS_H_
