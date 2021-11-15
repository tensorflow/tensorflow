/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"

namespace mlir {
namespace TF {
namespace {

// The name prefix of Flex ops.
constexpr char kFlexOpNamePrefix[] = "Flex";
constexpr char kDefaultMode[] = "DEFAULT";
constexpr char kLegacyIntegerMode[] = "LEGACY_INTEGER";

// Checks if the operation is allowlisted in the Legacy Integer mode.
bool IsAllowListedOpInLegacyMode(Operation *op) {
  if (llvm::isa<
          // clang-format off
          // go/keep-sorted start
          TF::AbsOp,
          TF::AddOp,
          TF::AddV2Op,
          TF::ArgMaxOp,
          TF::AvgPoolOp,
          TF::BiasAddOp,
          TF::BucketizeOp,
          TF::ConcatV2Op,
          TF::ConstOp,
          TF::Conv2DBackpropInputOp,
          TF::Conv2DOp,
          TF::DepthwiseConv2dNativeOp,
          TF::GatherV2Op,
          TF::IdentityOp,
          TF::MatMulOp,
          TF::MaxPoolOp,
          TF::MaximumOp,
          TF::MeanOp,
          TF::MinimumOp,
          TF::MulOp,
          TF::PadOp,
          TF::PadV2Op,
          TF::PartitionedCallOp,
          TF::Relu6Op,
          TF::ReluOp,
          TF::ReshapeOp,
          TF::SoftmaxOp,
          TF::StatefulPartitionedCallOp,
          TF::SubOp,
          TF::TransposeOp
          // go/keep-sorted end
          // clang-format on
          >(op)) {
    return true;
  }
  return false;
}

// Checks if the operation is allowlisted in the Default mode.
bool IsAllowListedOpInDefaultMode(Operation *op) {
  if (llvm::isa<
          // clang-format off
          // go/keep-sorted start
          TF::BiasAddOp,
          TF::ConstOp,
          TF::Conv2DBackpropInputOp,
          TF::Conv2DOp,
          TF::DepthwiseConv2dNativeOp,
          TF::IdentityOp,
          TF::MatMulOp,
          TF::PartitionedCallOp,
          TF::Relu6Op,
          TF::ReluOp,
          TF::StatefulPartitionedCallOp
          // go/keep-sorted end
          // clang-format on
          >(op)) {
    return true;
  }
  return false;
}

// Checks if the operation can be fused with bias.
inline bool IsFusibleWithBiasOp(Operation *op) {
  if (llvm::isa<
          // clang-format off
          TF::MatMulOp,
          TF::Conv2DOp,
          TF::DepthwiseConv2dNativeOp,
          TF::Conv2DBackpropInputOp,
          TF::Conv3DOp,
          TF::Conv3DBackpropInputV2Op
          // clang-format on
          >(op)) {
    return true;
  }
  return false;
}

// If the Add op can be fused as bias, converts it to BiasAdd op.
class ConvertAddToBiasAddPattern : public OpRewritePattern<TF::AddV2Op> {
 public:
  using OpRewritePattern<TF::AddV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::AddV2Op add_op,
                                PatternRewriter &rewriter) const override {
    // Fusable AddOp must have 1D constant value in the rhs since the BiasAddOp
    // can only accept 1D bias value and only constant value can be fused.
    DenseElementsAttr added_value;
    if (!matchPattern(add_op.y(), m_Constant(&added_value))) return failure();
    if (!IsFusibleWithBiasOp(add_op.x().getDefiningOp())) return failure();
    auto y_type = add_op.y().getType().template dyn_cast<RankedTensorType>();
    if (!y_type || y_type.getRank() != 1) return failure();

    rewriter.replaceOpWithNewOp<TF::BiasAddOp>(add_op, add_op.getType(),
                                               add_op.x(), add_op.y());
    return success();
  }
};

// If the Sub op can be fused as bias, converts it to BiasAdd op.
class ConvertSubToBiasAddPattern : public OpRewritePattern<TF::SubOp> {
 public:
  using OpRewritePattern<TF::SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SubOp sub_op,
                                PatternRewriter &rewriter) const override {
    // Fusable AddOp must have 1D constant value in the rhs since the BiasAddOp
    // can only accept 1D bias value and only constant value can be fused.
    DenseElementsAttr sub_value;
    if (!matchPattern(sub_op.y(), m_Constant(&sub_value))) return failure();
    if (!IsFusibleWithBiasOp(sub_op.x().getDefiningOp())) return failure();
    auto y_type = sub_op.y().getType().template dyn_cast<RankedTensorType>();
    if (!y_type || y_type.getRank() != 1) return failure();

    auto add_value = rewriter.create<TF::NegOp>(sub_op.getLoc(), sub_op.y());
    rewriter.replaceOpWithNewOp<TF::BiasAddOp>(sub_op, sub_op.getType(),
                                               sub_op.x(), add_value);
    return success();
  }
};

// Creates the custom option of the Flex ops.
inline void CreateFlexOpCustomOptions(const std::string &op_name,
                                      const std::string &node_def_str,
                                      std::string &custom_option_buffer) {
  auto flex_builder = absl::make_unique<flexbuffers::Builder>();
  flex_builder->Vector([&]() {
    flex_builder->String(op_name);
    flex_builder->String(node_def_str);
  });
  flex_builder->Finish();
  custom_option_buffer.assign(flex_builder->GetBuffer().begin(),
                              flex_builder->GetBuffer().end());
}

// Creates ElementsAttr for custom option.
inline OpaqueElementsAttr CustomOptionForFlexOp(OpBuilder *builder,
                                                const std::string &content) {
  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(content.size())}, builder->getIntegerType(8));
  return OpaqueElementsAttr::get(builder->getContext()->getLoadedDialect("tfl"),
                                 type,
                                 StringRef(content.data(), content.size()));
}

// Fallbacks ops that are not supported by TF Quantization to TFLite Flex ops.
class FallbackToFlexOps : public PassWrapper<FallbackToFlexOps, FunctionPass> {
 public:
  FallbackToFlexOps() {}
  explicit FallbackToFlexOps(const std::string &mode) { mode_ = mode; }
  FallbackToFlexOps(const FallbackToFlexOps &other) { mode_ = other.mode_; }

  void runOnFunction() override;

  StringRef getArgument() const final { return "quant-raise-flex-fallback"; }

  StringRef getDescription() const final {
    return "Fallback TF-Quantization-unsupported ops to TFLite Flex ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }

 private:
  // The mode of TF Quantization, might indicate different users/devices.
  Option<std::string> mode_{*this, "mode",
                            llvm::cl::desc("The mode of TF Quantization."),
                            llvm::cl::init("")};

  // Checks if the operation is allowlisted in the current mode.
  bool IsAllowListedOp(Operation *op) {
    if (mode_ == kDefaultMode) {
      return IsAllowListedOpInDefaultMode(op);
    } else if (mode_ == kLegacyIntegerMode) {
      return IsAllowListedOpInLegacyMode(op);
    } else {
      mlir::emitError(getFunction().getLoc(), "Unregconized mode: " + mode_);
      signalPassFailure();
      return true;
    }
  }

  // Converts the operation to a TFLite Flex op.
  bool ConvertToFlexOp(Operation *op);
};

bool FallbackToFlexOps::ConvertToFlexOp(Operation *op) {
  tensorflow::StatusOr<std::unique_ptr<tensorflow::NodeDef>> node_def =
      tensorflow::ConvertTFDialectOpToNodeDef(
          op, /*name=*/"", /*ignore_unregistered_attrs=*/true);
  if (!node_def.ok()) {
    op->emitError("Failed to obtain TensorFlow NodeDef: " +
                  node_def.status().ToString());
    return false;
  }
  std::string node_def_str;
  if (!(*node_def)->SerializeToString(&node_def_str)) {
    op->emitError("Failed to serialize tensorflow NodeDef");
    return false;
  }
  std::string op_name = (*node_def)->op();

  OpBuilder builder(op);
  std::string flex_op_name = std::string(kFlexOpNamePrefix) + op_name;
  std::string custom_option_buffer;
  CreateFlexOpCustomOptions(op_name, node_def_str, custom_option_buffer);
  auto flex_op = builder.create<TFL::CustomOp>(
      op->getLoc(), op->getResultTypes(), op->getOperands(), flex_op_name,
      CustomOptionForFlexOp(&builder, custom_option_buffer));
  op->replaceAllUsesWith(flex_op);
  op->erase();
  return true;
}

void FallbackToFlexOps::runOnFunction() {
  if (mode_.empty()) return;

  FuncOp func = getFunction();
  MLIRContext *ctx = &getContext();

  // Convert binary ops to BiasAdd ops if possible.
  OwningRewritePatternList patterns(ctx);
  patterns.insert<ConvertAddToBiasAddPattern, ConvertSubToBiasAddPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Convert unsupported ops to Flex ops.
  auto tf_dialect = ctx->getLoadedDialect<TF::TensorFlowDialect>();
  func.walk([&](Operation *op) {
    if (op->getDialect() != tf_dialect) return;
    if (IsAllowListedOp(op)) return;
    if (!ConvertToFlexOp(op)) signalPassFailure();
  });
}
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateFallbackToFlexOpsPass(
    const std::string &mode) {
  return std::make_unique<FallbackToFlexOps>(mode);
}

static PassRegistration<FallbackToFlexOps> pass([] {
  return CreateFallbackToFlexOpsPass(/*mode=*/kDefaultMode);
});

}  // namespace TF
}  // namespace mlir
