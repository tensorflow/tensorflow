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
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"

namespace mlir {
namespace quant {

// TODO(b/228928859): Improve the getter function to match attributes rather
// than function name.
std::unique_ptr<OpQuantSpec> GetTFOpQuantSpec(Operation* op) {
  auto spec = std::make_unique<OpQuantSpec>();
  if (auto call_op = dyn_cast<TF::PartitionedCallOp>(op)) {
    StringRef function_name =
        call_op.getFAttr().cast<FlatSymbolRefAttr>().getValue();
    if (!function_name.startswith("composite_")) {
      return spec;
    }
    if (function_name.contains("depthwise_conv2d")) {
      spec->coeff_op_quant_dim[1] = 3;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("conv2d")) {
      spec->coeff_op_quant_dim[1] = 3;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("matmul")) {
      spec->coeff_op_quant_dim[1] = -1;
      if (function_name.contains("with_bias") ||
          function_name.contains("and_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("einsum")) {
      spec->coeff_op_quant_dim[1] = -1;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("conv3d")) {
      spec->coeff_op_quant_dim[1] = 4;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("batch_matmul")) {
      spec->coeff_op_quant_dim[1] = -1;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("gather")) {
      // Note that gather has axis attribute that specifies channel axis.
      spec->coeff_op_quant_dim[0] = -1;
    }
    for (auto quantizable_operand : spec->coeff_op_quant_dim) {
      spec->quantizable_operands.insert(quantizable_operand.first);
    }
  }
  return spec;
}

std::unique_ptr<OpQuantScaleSpec> GetTfQuantScaleSpec(Operation* op) {
  auto scale_spec = std::make_unique<OpQuantScaleSpec>();
  if (llvm::isa<
          // clang-format off
          // go/keep-sorted start
          TF::AvgPoolOp,
          TF::ConcatOp,
          TF::ConcatV2Op,
          TF::ExpandDimsOp,
          TF::IdentityNOp,
          TF::IdentityOp,
          TF::MaxPoolOp,
          TF::PadV2Op,
          TF::RankOp,
          TF::ReshapeOp,
          TF::SelectOp,
          TF::SelectV2Op,
          TF::ShapeNOp,
          TF::ShapeOp,
          TF::SizeOp,
          TF::SqueezeOp,
          TF::TransposeOp
          // go/keep-sorted end
          // clang-format on
          >(op)) {
    scale_spec->has_same_scale_requirement = true;
  }
  return scale_spec;
}

}  // namespace quant
}  // namespace mlir
