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

#include "tensorflow/compiler/mlir/quantization/stablehlo/ops/stablehlo_op_quant_spec.h"

#include <memory>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::quant::stablehlo {

std::unique_ptr<OpQuantSpec> GetStableHloOpQuantSpec(Operation* op) {
  auto spec = std::make_unique<OpQuantSpec>();
  if (auto call_op = dyn_cast_or_null<TF::XlaCallModuleOp>(op)) {
    auto entry_function =
        call_op->getAttrOfType<FlatSymbolRefAttr>("_entry_function");
    StringRef function_name = entry_function.getValue();
    if (!function_name.starts_with("composite_")) {
      return spec;
    }
    if (function_name.contains("conv")) {
      spec->coeff_op_quant_dim[1] = 3;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("dot_general")) {
      spec->coeff_op_quant_dim[1] = -1;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("dot")) {
      spec->coeff_op_quant_dim[1] = -1;
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    }
    for (const auto [operand_idx, per_channel_dim] : spec->coeff_op_quant_dim) {
      spec->quantizable_operands.insert(operand_idx);
    }
  }
  return spec;
}

std::unique_ptr<OpQuantScaleSpec> GetStableHloQuantScaleSpec(Operation* op) {
  auto scale_spec = std::make_unique<OpQuantScaleSpec>();
  if (llvm::isa<mlir::stablehlo::BroadcastInDimOp,
                mlir::stablehlo::ConcatenateOp,
                mlir::stablehlo::DynamicReshapeOp, mlir::stablehlo::GatherOp,
                mlir::stablehlo::PadOp, mlir::stablehlo::ReduceWindowOp,
                mlir::stablehlo::ReshapeOp, mlir::stablehlo::SelectOp,
                mlir::stablehlo::SliceOp, mlir::stablehlo::TransposeOp>(op)) {
    scale_spec->has_same_scale_requirement = true;
  }
  return scale_spec;
}

bool IsOpQuantizableStableHlo(Operation* op) {
  if (isa<func::ConstantOp, mlir::stablehlo::ConstantOp>(op)) {
    // Constant ops do not have QuantizableResult attribute but can be
    // quantized.
    return true;
  } else if (op->hasTrait<OpTrait::IsTerminator>() ||
             isa<quantfork::QuantizeCastOp, quantfork::DequantizeCastOp>(op)) {
    // Terminators, qcast and decast are not quantizable.
    return false;
  }

  if (GetStableHloQuantScaleSpec(op)->has_same_scale_requirement) {
    return true;
  }

  const bool attr_enforced_quantizable =
      op->hasAttrOfType<StringAttr>(kQuantTraitAttrName) &&
      op->getAttrOfType<StringAttr>(kQuantTraitAttrName).getValue().str() ==
          QuantTraitValues[QuantizationTrait::FullyQuantizable];
  return attr_enforced_quantizable;
}

}  // namespace mlir::quant::stablehlo
