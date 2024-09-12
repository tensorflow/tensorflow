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

#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

// To be used with LLVM_DEBUG.
#define DEBUG_TYPE "stablehlo_opt_quant_spec"

namespace mlir::quant::stablehlo {
namespace {

using ::mlir::stablehlo::DotGeneralOp;
using ::stablehlo::quantization::Method;
using ::stablehlo::quantization::StaticRangePtq;

// Whether it represents a lifted function (i.e. `op` is the corresponding
// `XlaCallModuleOp`) that is explicitly marked `NoQuantization`.
bool IsDenylistedLiftedFunction(Operation* op) {
  if (auto xla_call_module_op = dyn_cast_or_null<TF::XlaCallModuleOp>(op);
      xla_call_module_op != nullptr) {
    absl::StatusOr<Method> method = GetQuantizationMethod(xla_call_module_op);
    if (method.ok() && method->has_no_quantization()) {
      return true;
    }
  }
  return false;
}

// Populates `spec.coeff_op_quant_dim` according to `xla_call_module_op`'s
// `_quantization_method` attribute. If there is an input `QuantizedType` with
// `dimension_specs` set, which represents the quantization dimension for the
// input, then the corresponding operand index -> quantization dimension mapping
// is set for `spec`.
// TODO: b/323478683 - Duplicate tracking of config will be eliminated.
// `OpQuantSpec` will be deprecated and `Method` will be used instead.
void PopulateCoeffOpQuantDimIfPerChannelQuantized(
    TF::XlaCallModuleOp xla_call_module_op, OpQuantSpec& spec) {
  absl::StatusOr<Method> method = GetQuantizationMethod(xla_call_module_op);
  if (method.ok() && method->has_static_range_ptq()) {
    // TODO: b/331145946 - Use `Method` accessors.
    const StaticRangePtq& static_range_ptq_spec = method->static_range_ptq();
    // Look for quantized dimension specs for each quantized type and
    // populate `coeff_op_quant_dim`.
    for (const auto& [operand_idx, quantized_type] :
         static_range_ptq_spec.input_quantized_types()) {
      if (quantized_type.has_dimension_specs()) {
        spec.coeff_op_quant_dim[operand_idx] =
            quantized_type.dimension_specs().dimension();
      }
    }
  }
}

}  // namespace

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
      // Looks up `Method` to see if it should be per-channel quantized and
      // populates the spec accordingly.
      PopulateCoeffOpQuantDimIfPerChannelQuantized(call_op, *spec);

      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("dot_general")) {
      const auto module_op = call_op->getParentOfType<ModuleOp>();

      const SymbolTable symbol_table(module_op);
      auto entry_func_op =
          dyn_cast_or_null<func::FuncOp>(symbol_table.lookup(function_name));
      auto dot_general_op = *entry_func_op.getOps<DotGeneralOp>().begin();
      if (auto optional_dim = GetDotGeneralQuantizationDim(dot_general_op);
          optional_dim) {
        spec->coeff_op_quant_dim[1] = optional_dim.value();
      } else {
        spec->coeff_op_quant_dim[1] = -1;
      }
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

std::unique_ptr<OpQuantScaleSpec> GetStableHloQuantConstraints(Operation* op) {
  auto scale_spec = std::make_unique<OpQuantScaleSpec>();
  if (llvm::isa<mlir::stablehlo::BroadcastInDimOp,
                mlir::stablehlo::ConcatenateOp,
                mlir::stablehlo::DynamicReshapeOp,
                mlir::stablehlo::DynamicSliceOp, mlir::stablehlo::GatherOp,
                mlir::stablehlo::PadOp, mlir::stablehlo::ReduceWindowOp,
                mlir::stablehlo::ReshapeOp, mlir::stablehlo::SelectOp,
                mlir::stablehlo::SliceOp, mlir::stablehlo::TransposeOp>(op)) {
    scale_spec->has_same_scale_requirement = true;
  }
  if (llvm::isa<mlir::stablehlo::DynamicSliceOp, mlir::stablehlo::GatherOp,
                mlir::stablehlo::PadOp, mlir::stablehlo::SliceOp>(op)) {
    scale_spec->has_same_operand_and_result_type_requirement = true;
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

  // `op` is not quantizable when it is an `XlaCallModuleOp` representing lifted
  // function whose `_quantization_method` attribute is marked `NoQuantization`.
  // This means this quantizable unit has been explicitly denylisted by the
  // user.
  if (IsDenylistedLiftedFunction(op)) {
    LLVM_DEBUG(llvm::errs() << "Denylisted quantizable unit: \n" << op << "\n");
    return false;
  }

  if (GetStableHloQuantConstraints(op)->has_same_scale_requirement) {
    return true;
  }

  const bool attr_enforced_quantizable =
      op->hasAttrOfType<StringAttr>(kQuantTraitAttrName) &&
      op->getAttrOfType<StringAttr>(kQuantTraitAttrName).getValue().str() ==
          QuantTraitValues[QuantizationTrait::FullyQuantizable];
  return attr_enforced_quantizable;
}

}  // namespace mlir::quant::stablehlo
