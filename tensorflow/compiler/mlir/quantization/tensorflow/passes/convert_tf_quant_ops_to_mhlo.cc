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
#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/ir/importexport/mangling.h"

namespace mlir {
namespace quant {
namespace {

class ConvertTFQuantOpsToMHLOPass
    : public PassWrapper<ConvertTFQuantOpsToMHLOPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTFQuantOpsToMHLOPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-convert-tf-quant-ops-to-mhlo";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Convert TF Quant ops to MHLO quantization";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
    registry.insert<mhlo::MhloDialect>();
    registry.insert<tf_type::TFTypeDialect>();
    registry.insert<quant::QuantizationDialect>();
  }

  void runOnOperation() override;
};

TensorType getSameShapeTensorType(TensorType tensor_type, Type element_type) {
  if (auto ranked_tensor_ty =
          tensor_type.dyn_cast_or_null<RankedTensorType>()) {
    return RankedTensorType::get(ranked_tensor_ty.getShape(), element_type);
  }
  if (auto unranked_tensor_ty =
          tensor_type.dyn_cast_or_null<UnrankedTensorType>()) {
    return UnrankedTensorType::get(element_type);
  }
  llvm_unreachable("unhandled type");
}

static PassRegistration<ConvertTFQuantOpsToMHLOPass> pass;

struct ReplaceConstDotHybridPattern : public RewritePattern {
  explicit ReplaceConstDotHybridPattern(MLIRContext *context)
      : RewritePattern(/*rootName=*/"tf.UniformQuantizedDotHybrid",
                       /*benefit=*/1, /*context=*/context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto this_op = cast<TF::UniformQuantizedDotHybridOp>(op);
    if (!this_op) return failure();

    // Check whether the constant op has qint8 element type.
    if (!getElementTypeOrSelf(this_op.rhs().getType()).isa<TF::Qint8Type>())
      return failure();

    // Check whether the rhs operand has constant op.
    OpaqueElementsAttr opaque_attr;
    if (!matchPattern(this_op.rhs(), m_Constant(&opaque_attr)))
      return failure();

    // Check whether the rhs_scales operand has constant op.
    DenseFPElementsAttr rhs_scales;
    if (!matchPattern(this_op.rhs_scales(), m_Constant(&rhs_scales)))
      return failure();

    // Check whether the rhs_zps operand has constant op.
    DenseIntElementsAttr rhs_zps;
    if (!matchPattern(this_op.rhs_zps(), m_Constant(&rhs_zps)))
      return failure();

    // Invalid quantization parameter.
    if (rhs_scales.size() != rhs_zps.size()) return failure();

    // Uniform Quantized type for the rhs.
    IntegerType storage_type = rewriter.getIntegerType(8);
    FloatType expressed_type = rewriter.getF32Type();
    int64_t storage_type_min = this_op.rhs_quantization_min_val();
    int64_t storage_type_max = this_op.rhs_quantization_max_val();
    int32_t quantized_dimension = this_op.rhs_quantization_axis();
    const unsigned flags = mlir::quant::QuantizationFlags::Signed;

    // Currently, PTQ supports per-tensor quantization, for now.
    if (quantized_dimension != -1) return failure();

    Type rhs_elem_ty;
    rhs_elem_ty = UniformQuantizedType::get(
        flags, storage_type, expressed_type, rhs_scales.getValues<float>()[0],
        rhs_zps.getValues<int32_t>()[0], storage_type_min, storage_type_max);

    Type rhs_type = getSameShapeTensorType(
        this_op.rhs().getType().cast<TensorType>(), rhs_elem_ty);

    llvm::StringRef mangled_tensor = opaque_attr.getValue();
    absl::string_view tensor_view(mangled_tensor.data(), mangled_tensor.size());
    tensorflow::TensorProto tensor_proto;
    tensorflow::Status status =
        tfg::mangling_util::DemangleTensor(tensor_view, &tensor_proto);
    if (!status.ok()) {
      return failure();
    }

    tensorflow::Tensor t;
    if (!t.FromProto(tensor_proto)) {
      return failure();
    }

    auto arr = t.flat<tensorflow::qint8>();
    auto new_opaque_attr = ElementsAttr(mlir::DenseElementsAttr::get(
        getSameShapeTensorType(rhs_type.cast<TensorType>(), storage_type),
        llvm::makeArrayRef(arr.data(), arr.size())));

    Value lhs = this_op.lhs();
    rewriter.setInsertionPointAfterValue(this_op.rhs());
    Value rhs = rewriter.create<mhlo::ConstantOp>(rewriter.getUnknownLoc(),
                                                  rhs_type, new_opaque_attr);

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mhlo::DotOp>(op, lhs, rhs,
                                             /*precision_config=*/nullptr);
    return success();
  }
};

void ConvertTFQuantOpsToMHLOPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp func = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.add<ReplaceConstDotHybridPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertTFQuantOpsToMHLOPass() {
  return std::make_unique<ConvertTFQuantOpsToMHLOPass>();
}

}  // namespace quant
}  // namespace mlir
