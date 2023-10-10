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

#include "xla/service/cpu_gpu_shape_verifier.h"

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"

namespace xla {

namespace {

bool HasInt4(const Shape& shape) {
  return ShapeUtil::HasPrimitiveType(shape, S4) ||
         ShapeUtil::HasPrimitiveType(shape, U4);
}

Status VerifyS4U4Usage(HloInstruction* instruction) {
  if (HasInt4(instruction->shape())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "S4/U4 is currently not support on XLA CPU/GPU, but got instruction "
        " with S4/U4 output: %s",
        instruction->ToString()));
  }
  return OkStatus();
}
}  // namespace

Status CpuGpuShapeVerifier::Preprocess(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      hlo->shape(), [&](const Shape& shape, const ShapeIndex&) {
        if (shape.has_layout()) {
          if (LayoutUtil::IsSparseArray(shape)) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "The XLA CPU/GPU backend does not support sparse shapes: %s",
                hlo->ToString()));
          }
          if (shape.layout().element_size_in_bits() != 0) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "The XLA CPU/GPU backend does not support custom element "
                "sizes: %s",
                hlo->ToString()));
          }
        }
        return OkStatus();
      }));

  TF_RETURN_IF_ERROR(VerifyS4U4Usage(hlo));
  return ShapeVerifier::Preprocess(hlo);
}

}  // namespace xla
