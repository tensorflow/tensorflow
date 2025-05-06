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

#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

// TODO - b/296503614: [Converter Component][TF-Quantizer] Reflect custom traits
// from TF-Quantizer to stableHLO quantization
bool IsOpWithDataMovementTrait(Operation* op) {
  // Supported data movement ops. These ops do not perform any computations and
  // has one result operand.
  return isa<TF::IdentityOp, TF::CastOp, TF::ReshapeOp, TF::XlaShardingOp,
             TF::GatherOp, TF::GatherV2Op, TF::XlaGatherOp, TF::ExpandDimsOp,
             TF::SqueezeOp, TF::TransposeOp>(op);
}

bool IsOpWithQuantizableTrait(Operation* op) {
  // Supported quantizable ops.
  return isa<TF::XlaConvV2Op, TF::XlaDotV2Op, TF::MatMulOp, TF::Conv2DOp,
             TF::GatherOp, TF::GatherV2Op, TF::XlaGatherOp,
             TF::ResourceGatherOp, TF::DepthwiseConv2dNativeOp, TF::Conv3DOp,
             TF::BatchMatMulV2Op, TF::EinsumOp>(op);
}

bool IsOpWithInt8TypeOperand(Operation* op) {
  return (isa<TF::XlaConvV2Op, TF::XlaDotV2Op, TF::XlaGatherOp, TF::GatherOp,
              TF::GatherV2Op>(op));
}

bool IsValueWithQuantizablePrecision(Value val) {
  auto type = mlir::dyn_cast<ShapedType>(val.getType());
  if (!type) return false;
  // Supported original tensor data types.
  if (type.getElementType().isF32() || type.getElementType().isBF16())
    return true;
  return false;
}

std::optional<tensorflow::quantization::QuantizationComponentSpec>
GetWeightComponentSpec(
    const tensorflow::quantization::QuantizationOptions& quantization_options) {
  for (auto& cur_spec : quantization_options.quantization_method()
                            .quantization_component_specs()) {
    if (cur_spec.quantization_component() ==
        tensorflow::quantization::QuantizationComponentSpec::COMPONENT_WEIGHT)
      return cur_spec;
  }
  return std::nullopt;
}

// TODO(b/228928859): Improve the getter function to match attributes rather
// than function name.
std::unique_ptr<OpQuantSpec> GetTFOpQuantSpec(Operation* op) {
  auto spec = std::make_unique<OpQuantSpec>();
  if (auto call_op = dyn_cast<TF::PartitionedCallOp>(op)) {
    StringRef function_name =
        mlir::cast<FlatSymbolRefAttr>(call_op.getFAttr()).getValue();
    if (!function_name.starts_with("composite_")) {
      return spec;
    }
    if (function_name.contains("depthwise_conv2d")) {
      spec->coeff_op_quant_dim[1] = 3;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1}, GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("conv2d")) {
      spec->coeff_op_quant_dim[1] = 3;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1}, GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("matmul")) {
      spec->coeff_op_quant_dim[1] = -1;
      if (function_name.contains("with_bias") ||
          function_name.contains("and_bias")) {
        spec->biases_params[2] = {{0, 1}, GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("einsum")) {
      spec->coeff_op_quant_dim[1] = -1;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1}, GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("conv3d")) {
      spec->coeff_op_quant_dim[1] = 4;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1}, GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("batch_matmul")) {
      spec->coeff_op_quant_dim[1] = -1;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1}, GetUniformQuantizedTypeForBias};
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
