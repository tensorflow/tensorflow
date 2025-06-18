/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_lift_as_function_call.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/tf_passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::tf_quant::stablehlo {

#define GEN_PASS_DEF_INSERTWEIGHTPARAMPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/tf_passes.h.inc"

namespace {

using ::stablehlo::quantization::Method;
using ::stablehlo::quantization::QuantizedType;
using ::stablehlo::quantization::WeightOnlyPtq;

// Inserts quantization parameters of weights for weight-only quantization and
// dynamic range quantization of `stablehlo.convolution` and
// `stablehlo.dot_general`.
class InsertWeightParamPass
    : public impl::InsertWeightParamPassBase<InsertWeightParamPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertWeightParamPass)

  using impl::InsertWeightParamPassBase<
      InsertWeightParamPass>::InsertWeightParamPassBase;

 private:
  void runOnOperation() override;
};

// Inserts quantization parameters for weights for hybrid quantization of
// `stablehlo.convolution` and `stablehlo.dot_general`.
class InsertWeightParamPattern
    : public OpTraitRewritePattern<OpTrait::ConstantLike> {
 public:
  explicit InsertWeightParamPattern(MLIRContext* context)
      : OpTraitRewritePattern(context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (op->getNumResults() != 1) {
      return failure();
    }
    auto type = mlir::cast<TensorType>(op->getResult(0).getType());
    if (!type || !type.getElementType().isF32()) {
      return failure();
    }
    if (!op->hasOneUse() ||
        !IsWeightQuantizableFunction(*op->getUses().begin(), type.getRank())) {
      return failure();
    }

    Operation* quantizable_op = *op->getUsers().begin();
    DenseFPElementsAttr attr;
    matchPattern(op->getResult(0), m_Constant(&attr));

    Method method = GetQuantizationMethodOrDefault(quantizable_op);
    const WeightOnlyPtq& weight_only_ptq = method.weight_only_ptq();

    Type weight_type;
    if (IsPerTensor(weight_only_ptq)) {
      weight_type =
          dyn_cast<quant::QuantizedType>(GetUniformQuantizedTypeForWeight(
              attr, /*symmetric=*/true, /*num_bits=*/8, /*is_signed=*/true,
              /*narrow_range=*/true, /*legacy_float_scale=*/false));
    } else {
      int quantization_dimension = GetQuantizationDimension(
          weight_only_ptq, cast<TF::XlaCallModuleOp>(quantizable_op));
      weight_type = GetUniformQuantizedPerAxisTypeForWeight(
          attr, quantization_dimension, /*symmetric=*/true, /*num_bits=*/8,
          /*is_signed=*/true,
          /*narrow_range=*/true, /*legacy_float_scale=*/false);
    }

    auto quant_type = dyn_cast<quant::QuantizedType>(weight_type);
    if (!quant_type) {
      op->emitError(
          "Failed to get weight quantization parameters for weight-only "
          "quantization.");
      return failure();
    }

    const Type expressed_type = op->getResult(0).getType();
    const Type quantized_type =
        quant_type.castFromExpressedType(expressed_type);

    rewriter.setInsertionPointAfter(op);
    auto q = rewriter.create<mlir::quant::ir::QuantizeCastOp>(
        op->getLoc(), quantized_type, op->getResult(0));
    auto dq = rewriter.create<mlir::quant::ir::DequantizeCastOp>(
        op->getLoc(), expressed_type, q);
    quantizable_op->setOperand(1, dq.getResult());
    return success();
  }

  // Checks if the operand is second operand of `tf.XlaCallModule` op for
  // `stablehlo.convolution` or `stablehlo.dot_general` with fully_quantizable
  // trait.
  static bool IsWeightQuantizableFunction(OpOperand& operand, int64_t rank) {
    if (operand.getOperandNumber() != 1) {
      return false;
    }
    Operation* user = operand.getOwner();
    if (!IsWeightOnlyQuantizableOp(*user)) {
      return false;
    }
    Method method = GetQuantizationMethodOrDefault(user);
    return HasValidWeightOnlyPtqMethod(method.weight_only_ptq(), rank);
  }

 private:
  static bool HasValidWeightOnlyPtqMethod(const WeightOnlyPtq& weight_only_ptq,
                                          int64_t rank) {
    const auto& input_quantized_types = weight_only_ptq.input_quantized_types();
    if (IsPerTensor(weight_only_ptq)) {
      return true;
    }
    // `input_quantized_types` should contain spec for quantization type of the
    // second operand, which is weight.
    const QuantizedType& quantized_type = input_quantized_types.at(1);
    if (const auto& specs = quantized_type.dimension_specs();
        specs.has_dimension()) {
      return specs.dimension() >= 0 && specs.dimension() < rank;
    }
    return true;
  }

  static bool IsPerTensor(const WeightOnlyPtq& weight_only_ptq) {
    const auto& input_quantized_types = weight_only_ptq.input_quantized_types();
    if (input_quantized_types.empty()) {
      return true;
    }
    auto weight_type = input_quantized_types.find(1);
    if (weight_type == input_quantized_types.end()) {
      return true;
    }
    return weight_type->second.has_per_tensor();
  }

  static int GetQuantizationDimension(const WeightOnlyPtq& weight_only_ptq,
                                      TF::XlaCallModuleOp op) {
    const QuantizedType& quantized_type =
        weight_only_ptq.input_quantized_types().at(1);
    if (quantized_type.dimension_specs().has_dimension()) {
      return quantized_type.dimension_specs().dimension();
    }
    return GetDefaultQuantizationDimension(op);
  }

  // Determines quantization dimension of weights for given `tf.XlaCallModule`
  // op. For convolution, returns output feature dimension of the kernel. For
  // dot_general, returns the first non-contracting dimension, non-batching
  // dimension. If such dimension does not exists, returns the last dimension of
  // rhs.
  static int64_t GetDefaultQuantizationDimension(TF::XlaCallModuleOp op) {
    const StringRef function_name = GetEntryFunctionName(op);
    const auto module_op = op->getParentOfType<ModuleOp>();
    const SymbolTable symbol_table(module_op);
    func::FuncOp func = symbol_table.lookup<func::FuncOp>(function_name);

    if (function_name.contains("conv")) {
      return (*(func.getOps<mlir::stablehlo::ConvolutionOp>().begin()))
          .getDimensionNumbers()
          .getKernelOutputFeatureDimension();
    } else if (function_name.contains("dot_general")) {
      auto dot = *(func.getOps<mlir::stablehlo::DotGeneralOp>().begin());
      const ::mlir::stablehlo::DotDimensionNumbersAttr dimension_numbers =
          dot.getDotDimensionNumbers();
      ArrayRef<int64_t> rhs_contracting_dims =
          dimension_numbers.getRhsContractingDimensions();
      ArrayRef<int64_t> rhs_batching_dims =
          dimension_numbers.getRhsBatchingDimensions();
      int64_t rank = cast<TensorType>(dot.getRhs().getType()).getRank();
      for (int i = 0; i < rank; ++i) {
        // Return the first non-contracting, non-batching dimension of rhs.
        if (llvm::find(rhs_contracting_dims, i) == rhs_contracting_dims.end() &&
            llvm::find(rhs_batching_dims, i) == rhs_batching_dims.end()) {
          return i;
        }
      }
    }
    return cast<TensorType>(op.getOperand(1).getType()).getRank() - 1;
  }
};

void InsertWeightParamPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* context = func.getContext();
  RewritePatternSet patterns(context);

  patterns.add<InsertWeightParamPattern>(context);

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

}  // namespace mlir::tf_quant::stablehlo
