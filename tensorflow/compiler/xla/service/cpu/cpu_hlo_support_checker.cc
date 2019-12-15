/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_hlo_support_checker.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

StatusOr<bool> CpuHloSupportChecker::Run(HloModule* module) {
  for (auto* computation : module->computations()) {
    for (const auto& instruction : computation->instructions()) {
      TF_RETURN_IF_ERROR(
          ShapeUtil::ValidateShapeWithOptionalLayout(instruction->shape()));
      TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
          instruction->shape(),
          [&instruction](const Shape& subshape, const ShapeIndex&) {
            if (LayoutUtil::IsSparseArray(subshape)) {
              return xla::Unimplemented(
                  "CPU backend does not support HLO instruction %s with shape "
                  "containing a sparse layout: %s",
                  instruction->ToString(),
                  ShapeUtil::HumanStringWithLayout(instruction->shape()));
            }
            return Status::OK();
          }));
    }
  }
  return false;
}

}  // namespace xla
