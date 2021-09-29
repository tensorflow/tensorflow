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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCTION_DIMENSION_GROUPER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCTION_DIMENSION_GROUPER_H_

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Groups adjacent (logically and physically) reduced dimensions in reduction
// input.
//
// Precondition: ReductionLayoutNormalizer has been run (physical proximity and
// logical proximity become the same).
//
// For example,
//
//   f[] out = reduce(f[10,20,30] input, dimensions={0,1,2})
//
// becomes:
//
//   f[600] tmp = f[600] bitcast(f[10,20,30] input)
//   f[] out = reduce(f[600] tmp, dimensions={0})
//
class ReductionDimensionGrouper : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "reduction-dimension-grouper";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCTION_DIMENSION_GROUPER_H_
