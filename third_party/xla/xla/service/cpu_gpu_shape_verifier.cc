/* Copyright 2023 The OpenXLA Authors.

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
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"

namespace xla {

absl::Status CpuGpuShapeVerifier::Preprocess(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      hlo->shape(), [&](const Shape& shape, const ShapeIndex&) {
        if (shape.has_layout()) {
          if (!primitive_util::IsSubByteNonPredType(shape.element_type()) &&
              shape.layout().element_size_in_bits() != 0) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "The XLA CPU/GPU backend does not support custom element sizes "
                "on non-sub-byte types: %s",
                hlo->ToString()));
          }
        }
        return absl::OkStatus();
      }));

  return ShapeVerifier::Preprocess(hlo);
}

}  // namespace xla
