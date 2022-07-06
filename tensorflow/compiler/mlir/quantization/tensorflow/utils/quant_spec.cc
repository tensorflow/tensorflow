/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/quant_spec.h"

#include <memory>

#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

std::unique_ptr<OpQuantScaleSpec> GetTfQuantScaleSpec(Operation* op) {
  auto scale_spec = std::make_unique<OpQuantScaleSpec>();
  if (llvm::isa<
          // clang-format off
          // go/keep-sorted start
          TF::ConcatV2Op,
          TF::IdentityOp,
          TF::MaxPoolOp,
          TF::PadV2Op,
          TF::ReshapeOp,
          TF::SqueezeOp
          // go/keep-sorted end
          // clang-format on
          >(op)) {
    scale_spec->has_same_scale_requirement = true;
  }
  return scale_spec;
}

}  // namespace quant
}  // namespace mlir
