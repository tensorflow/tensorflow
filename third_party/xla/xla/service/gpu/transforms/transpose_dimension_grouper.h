/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_TRANSFORMS_TRANSPOSE_DIMENSION_GROUPER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_TRANSPOSE_DIMENSION_GROUPER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Groups dimensions that are adjacent (logically and physically) in the
// transpose operand and the transpose output.
//
// Precondition: LayoutNormalization has been run (physical proximity and
// logical proximity become the same).
//
// For example,
//
//   out = f32[30,10,20] transpose(f32[10,20,30] input, dimensions={2,0,1})
//
// becomes:
//
//   tmp = f32[200,30] bitcast(f32[10,20,30] input)
//   transpose = f32[30,200] transpose(f32[200,30] tmp, dimensions={1,0})
//   out = f32[30,0,20] bitcast(f32[30,200] transpose)
//
class TransposeDimensionGrouper : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "transpose-dimension-grouper";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_TRANSPOSE_DIMENSION_GROUPER_H_
