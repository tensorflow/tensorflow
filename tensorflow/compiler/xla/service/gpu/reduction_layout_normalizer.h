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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCTION_LAYOUT_NORMALIZER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCTION_LAYOUT_NORMALIZER_H_

#include <optional>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Enforces default (minor-to-major) layout on all reduction inputs.
// Note that since reduction output can request a custom layout,
// this pass only guarantees standard layout for the input.
//
// For example,
//
//   f[20,30]{0,1} out = reduce(f[10,20,30]{2,0,1} input, dimensions={0})
//
// becomes:
//
//   f[20,10,30] tmp = f[20,10,30] bitcast(f[10,20,30]{2,0,1} input)
//   f[20,30]{0,1} out = reduce(f[20,10,30]{2,1,0} tmp, dimensions={1})
class ReductionLayoutNormalizer : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "reduction-layout-normalizer";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCTION_LAYOUT_NORMALIZER_H_
