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

#include "tensorflow/compiler/xla/service/gpu/cublas_padding_requirements.h"

#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

namespace {

bool DimensionRequiresPadding(const int64_t size, const PrimitiveType data_type,
                              const se::CudaComputeCapability cc) {
  for (const CublasPaddingRequirement& requirement :
       CublasPaddingRequirements) {
    if (cc.IsAtLeast(requirement.min_compute_capability) &&
        data_type == requirement.data_type &&
        size % requirement.multiple_of != 0) {
      return true;
    }
  }
  return false;
}

bool ShapeRequiresPadding(const Shape& shape,
                          const se::CudaComputeCapability cc) {
  // Since dots are canonicalized before padding only the last two dimensions
  // of each operand represent non-batch dimensions and may need padding.
  return DimensionRequiresPadding(shape.dimensions(shape.rank() - 1),
                                  shape.element_type(), cc) ||
         DimensionRequiresPadding(shape.dimensions(shape.rank() - 2),
                                  shape.element_type(), cc);
}

}  // namespace

bool CublasRequiresPadding(const HloDotInstruction& dot,
                           const se::CudaComputeCapability cc) {
  return ShapeRequiresPadding(dot.operand(0)->shape(), cc) ||
         ShapeRequiresPadding(dot.operand(1)->shape(), cc);
}

}  // namespace gpu
}  // namespace xla
