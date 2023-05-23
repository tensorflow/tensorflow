/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DOT_DIMENSION_SORTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DOT_DIMENSION_SORTER_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Sorts contracting dimensions of dot() operands when this reduces the
// number of transposes. Example:
// dot(p0, p1), lhs_contracting_dims={3,2}, rhs_contracting_dims={2,1}  ->
// dot(p0, p1), lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
// The first case gets transposes inserted by dot_decomposer, the second one
// does not and thus is generally more efficient.

// TODO(b/265688934): do the same for batch dimensions?

class DotDimensionSorter : public HloModulePass {
 public:
  absl::string_view name() const override { return "dot_dimension_sorter"; }

  // Run the pass on computations in 'module'.
  // Returns whether the 'module' was changed.
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DOT_DIMENSION_SORTER_H_
