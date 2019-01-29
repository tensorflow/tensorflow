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

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// This pass opportunistically finds input and output buffers that can be
// aliased, and writes the alias config into the HloModule.
//
// The input and the output buffers can be in any shape, and each output buffer
// can alias with an input buffer with the same size. Each input buffer may only
// alias with a single output buffer. For example, for the following parameter
// and the output buffers,
//
//  Parameters : { P1(2MiB), P2(4MiB), P3(8MiB), P4(4MiB), P5(4MiB), ... }
//  Outputs    : { O1(4MiB), O2(2MiB), O3(4MiB), O4(6MiB), O5(4MiB), ... }
//
// one potential aliasing would be (O1, P2), (O2, P1), (O3, P4), (O5, P5), ..
class OptimizeInputOutputBufferAlias : public HloModulePass {
  using ShapeSizeFunction = std::function<int64(const Shape&)>;

 public:
  OptimizeInputOutputBufferAlias(ShapeSizeFunction size_func)
      : size_func_(size_func) {}
  ~OptimizeInputOutputBufferAlias() override = default;

  absl::string_view name() const override {
    return "optimize_input_output_buffer_alias.h";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  friend class OptimizeInputOutputBufferAliasTest;

  StatusOr<bool> Build(const Shape& input_shape, const Shape& output_shape,
                       HloInputOutputAliasConfig* alias_config);
  ShapeSizeFunction size_func_ = nullptr;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_OPTIMIZE_INPUT_OUTPUT_BUFFER_ALIAS_H_
