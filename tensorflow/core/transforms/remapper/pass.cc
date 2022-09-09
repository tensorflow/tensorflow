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

#include "tensorflow/core/transforms/remapper/pass.h"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "mlir/Dialect/PDL/IR/PDL.h"  // from @llvm-project
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/transforms/remapper/remapping_helper.h"
#include "tensorflow/core/transforms/utils/pdll/utils.h"
#include "tensorflow/core/transforms/utils/utils.h"
#include "tensorflow/core/util/util.h"

namespace mlir {
namespace tfg {
namespace mkl {
#include "tensorflow/core/transforms/remapper/pdll/MklPDLLPatterns.h.inc"
}  // namespace mkl

#define GEN_PASS_DEF_REMAPPER
#include "tensorflow/core/transforms/passes.h.inc"

// Convert Sigmoid+Mul to Swish
// Mul(x, Sigmoid(x)) --> _MklSwish(x)
class MatchMulSigmoid : public RewritePattern {
 public:
  explicit MatchMulSigmoid(MLIRContext *context)
      : RewritePattern("tfg.Mul", PatternBenefit(/*benefit=*/1), context),
        sigmoid_name_("tfg.Sigmoid", context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    TypeAttr dtype_attr = op->getAttrOfType<TypeAttr>("T");
    if (!dtype_attr.getValue().isa<Float32Type>() &&
        !dtype_attr.getValue().isa<BFloat16Type>()) {
      return failure();
    }

    if (!util::OpHasDevice(op, tensorflow::DEVICE_CPU)) return failure();

    TFOp mul_wrapper(op);

    Value sigmoid = op->getOperand(0);
    Value x = op->getOperand(1);

    auto sigmoidOperandEqToX = [&](Value sigmoid, Value x) {
      Operation *op = sigmoid.getDefiningOp();
      return op && op->getName() == sigmoid_name_ && op->getOperand(0) == x;
    };

    if (!sigmoidOperandEqToX(sigmoid, x)) {
      // The operands are commutative and it may have both sigmoid operands.
      // Swap them then check it again.
      std::swap(sigmoid, x);
      if (!sigmoidOperandEqToX(sigmoid, x)) return failure();
    }

    SmallVector<Value> operands;
    // Set up non-control operand.
    operands.push_back(x);
    // Control operands come after regular operands.
    llvm::append_range(operands, mul_wrapper.getControlOperands());

    Operation *new_op =
        rewriter.create(op->getLoc(), rewriter.getStringAttr("tfg._MklSwish"),
                        operands, op->getResultTypes(), op->getAttrs());
    rewriter.replaceOp(op, new_op->getResults());

    return success();
  }

 private:
  // This is used to eliminate the string comparison by caching the handle of an
  // operation name.
  OperationName sigmoid_name_;
};

// This enum class is used as a template parameter and meant for alias to tfg op
// name.
// TODO(intel-tf): Add more items as needed.
enum class OpKind { Relu, Relu6, Elu, LeakyRelu, Tanh, Sigmoid };

inline std::string GetTfgOpName(OpKind op_kind) {
  switch (op_kind) {
    case OpKind::Relu:
      return "tfg.Relu";
    case OpKind::Relu6:
      return "tfg.Relu6";
    case OpKind::Elu:
      return "tfg.Elu";
    case OpKind::LeakyRelu:
      return "tfg.LeakyRelu";
    case OpKind::Tanh:
      return "tfg.Tanh";
    case OpKind::Sigmoid:
      return "tfg.Sigmoid";
    default:
      return "tfg.NoOp";
  }
}

class RemapperPatternBase : public RewritePattern {
 public:
  RemapperPatternBase(StringRef opName, OpPropertyHelper &helper,
                      PatternBenefit benefit = PatternBenefit(1))
      : RewritePattern(opName, benefit, helper.getDialect()->getContext()),
        helper_(helper) {}
  RemapperPatternBase(MatchAnyOpTypeTag tag, OpPropertyHelper &helper,
                      PatternBenefit benefit = PatternBenefit(1))
      : RewritePattern(tag, benefit, helper.getDialect()->getContext()),
        helper_(helper) {}

 protected:
  OpPropertyHelper helper_;
};

static std::unique_ptr<OperationState> GetContractionBiasAddOpState(
    OpBuilder &builder, const OpPropertyHelper &helper,
    Operation *contraction_op, Operation *bias_add_op) {
  // Fused op name dependes on original contraction op name.
  std::string fused_op_name;
  if (helper.getDialect()->IsConv2D(contraction_op)) {
    fused_op_name = "tfg._FusedConv2D";
  } else if (helper.getDialect()->IsMatMul(contraction_op)) {
    fused_op_name = "tfg._FusedMatMul";
  } else if (helper.getDialect()->IsDepthwiseConv2dNative(contraction_op)) {
    fused_op_name = "tfg._FusedDepthwiseConv2dNative";
  } else if (helper.getDialect()->IsConv3D(contraction_op)) {
    fused_op_name = "tfg._FusedConv3D";
  } else {
    return nullptr;
  }

  SmallVector<Location> fused_locs{contraction_op->getLoc(),
                                   bias_add_op->getLoc()};
  auto state = std::make_unique<OperationState>(builder.getFusedLoc(fused_locs),
                                                fused_op_name);
  SmallVector<Value> operands;
  Value input = contraction_op->getOperand(0);
  Value filter = contraction_op->getOperand(1);
  Value bias = bias_add_op->getOperand(1);
  operands.push_back(input);
  operands.push_back(filter);
  operands.push_back(bias);
  state->addOperands(operands);
  state->addOperands(TFOp(contraction_op).getControlOperands());
  state->addOperands(TFOp(bias_add_op).getControlOperands());
  state->addTypes(bias_add_op->getResultTypes());
  state->attributes = contraction_op->getAttrs();
  state->attributes.set("fused_ops", builder.getStrArrayAttr({"BiasAdd"}));
  state->attributes.set("num_args", builder.getI32IntegerAttr(1));
  // Default values for epsilon and leakyrelu_alpha
  state->attributes.set("epsilon", builder.getF32FloatAttr(0.0001));
  state->attributes.set("leakyrelu_alpha", builder.getF32FloatAttr(0.2));
  return state;
}

// Contraction + BiasAdd
// TODO(intel-tf): Support Contraction + {Add, AddV2} fusion in the case it has
// similar semantic of contraction + BiasAdd
class ContractionBiasAddRewriter : public RemapperPatternBase {
 public:
  explicit ContractionBiasAddRewriter(OpPropertyHelper &helper)
      : RemapperPatternBase("tfg.BiasAdd", helper, PatternBenefit(1)) {}

  // Constructor used by derived pattern rewritter class that may have
  // different root operation name. Currently, pattern is
  // matched from root op to its inputs.
  explicit ContractionBiasAddRewriter(StringRef op_name,
                                      OpPropertyHelper &helper,
                                      PatternBenefit benefit)
      : RemapperPatternBase(op_name, helper, benefit) {}

  using Pattern = ContractionBiasAdd;

  bool matchPattern(Operation *op, Pattern &pattern) const {
    // Not allowing control flow on BiasAdd
    if (helper_.HasControlOperandsOrResultUsers(op)) return false;
    Operation *contraction_op = op->getOperand(0).getDefiningOp();
    if (!helper_.IsContraction(contraction_op) ||
        helper_.HasControlOperandsOrResultUsers(contraction_op) ||
        !helper_.HaveSameDataType(op, contraction_op) ||
        !helper_.HasAtMostOneUserOfResult0(contraction_op)) {
      return false;
    }
    pattern.contraction = contraction_op;
    pattern.bias_add = op;
    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Pattern pattern;
    if (!matchPattern(op, pattern)) return failure();
    if (!helper_.IsDeviceCompatible(pattern)) return failure();
    std::unique_ptr<OperationState> state = GetContractionBiasAddOpState(
        rewriter, helper_, pattern.contraction, pattern.bias_add);
    Operation *fused_op = rewriter.create(*state);
    TFOp(fused_op).setName(TFOp(op).nameAttr());
    rewriter.replaceOp(op, fused_op->getResults());
    return success();
  }
};

// BasePattern + Activation
template <typename BasePatternRewriter, OpKind activation>
class BasePatternActivationRewriter : public BasePatternRewriter {
 public:
  explicit BasePatternActivationRewriter(OpPropertyHelper &helper)
      : BasePatternRewriter(GetTfgOpName(activation), helper,
                            PatternBenefit(1)) {}

  using BasePattern = typename BasePatternRewriter::Pattern;
  using Pattern = std::conditional_t<
      std::is_same<BasePatternRewriter, ContractionBiasAddRewriter>::value,
      ContractionBiasAddActivation, void>;

  bool matchPattern(Operation *op, BasePattern &base_pattern,
                    Pattern &pattern) const {
    // Although template instantiation guarantuees that only valid activation is
    // set as the root operation, a sanity check is added here.
    if (this->helper_.getDialect()->IsNoOp(op)) return false;
    if (this->helper_.HasControlOperandsOrResultUsers(op)) return false;

    // TODO(intel-tf): Add support for more patterns.
    if constexpr (std::is_same<BasePattern, ContractionBiasAdd>::value &&
                  std::is_same<Pattern, ContractionBiasAddActivation>::value) {
      Operation *bias_add_op = op->getOperand(0).getDefiningOp();
      if (!this->helper_.getDialect()->IsBiasAdd(bias_add_op) ||
          !this->helper_.HaveSameDataType(op, bias_add_op) ||
          !this->helper_.HasAtMostOneUserOfResult0(bias_add_op)) {
        return false;
      }
      if (!BasePatternRewriter::matchPattern(bias_add_op, base_pattern)) {
        return false;
      }
      pattern.contraction = base_pattern.contraction;
      pattern.bias_add = base_pattern.bias_add;
      pattern.activation = op;
      return true;
    }

    return false;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    BasePattern base_pattern;
    Pattern pattern;
    if (!matchPattern(op, base_pattern, pattern)) return failure();
    if constexpr (!std::is_same<BasePatternRewriter,
                                ContractionBiasAddRewriter>::value) {
      return failure();
    }
    if (!this->helper_.IsDeviceCompatible(pattern)) return failure();
    Operation *&contraction_op = pattern.contraction;
    Operation *&bias_add_op = pattern.bias_add;
    Operation *&activation_op = pattern.activation;
    const std::string activation_op_name =
        activation_op->getName().stripDialect().str();
    // Currently, supported activations are:
    //    _FusedMatMul: Relu, Relu6, Elu, LeakyRelu, Tanh, and Sigmoid
    //    _Fused*Conv*: Relu, Relu6, Elu and LeakyRelu
    if ((activation_op_name == "Tanh" || activation_op_name == "Sigmoid") &&
        !this->helper_.getDialect()->IsMatMul(contraction_op)) {
      return failure();
    }

    std::unique_ptr<OperationState> state = GetContractionBiasAddOpState(
        rewriter, this->helper_, contraction_op, bias_add_op);
    SmallVector<Location> fused_locs{state->location, activation_op->getLoc()};
    state->location = rewriter.getFusedLoc(fused_locs);
    state->attributes.set(
        "fused_ops", rewriter.getStrArrayAttr({"BiasAdd", activation_op_name}));
    if (this->helper_.getDialect()->IsLeakyRelu(activation_op)) {
      state->attributes.set("leakyrelu_alpha", activation_op->getAttr("alpha"));
    }
    Operation *fused_op = rewriter.create(*state);
    TFOp(fused_op).setName(TFOp(op).nameAttr());
    rewriter.replaceOp(op, fused_op->getResults());
    return success();
  }
};

template <template <OpKind> class PatternT, OpKind... op_kinds,
          typename... Args>
static void InsertPatterns(RewritePatternSet &patterns, Args &&...args) {
  patterns.insert<PatternT<op_kinds>...>(std::forward<Args>(args)...);
}

template <OpKind activation>
using ContractionBiasAddActivationRewriter =
    BasePatternActivationRewriter<ContractionBiasAddRewriter, activation>;

class Remapper : public impl::RemapperBase<Remapper> {
 public:
  Remapper() = default;
  explicit Remapper(bool enable_onednn_patterns, bool xla_auto_clustering) {
    enable_onednn_patterns_ = enable_onednn_patterns;
    xla_auto_clustering_ = xla_auto_clustering;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pdl::PDLDialect, pdl_interp::PDLInterpDialect>();
  }

  LogicalResult initialize(MLIRContext *context) override {
    helper_ = OpPropertyHelper(context->getOrLoadDialect<TFGraphDialect>(),
                               enable_onednn_patterns_, xla_auto_clustering_);
    RewritePatternSet patterns(context);
    populateRemapperPatterns(context, patterns);
    RegisterPDLLUtils(patterns);
    final_patterns_ = std::move(patterns);
    return success();
  }

  void runOnOperation() override;

 private:
  void populateRemapperPatterns(MLIRContext *context,
                                RewritePatternSet &patterns) {
    if (verify_pdll_patterns_only_) {
      populateRemapperPDLLPatterns(patterns);
      return;
    }
    if (enable_onednn_patterns_) {
      patterns.insert<MatchMulSigmoid>(context);
      // TODO(chiahungduan): Currently, the only pattern implemented in PDLL is
      // the same one as `MatchMulSigmoid`. Remove the one of them when there's
      // a decision that which one is preferred.
      populateRemapperPDLLPatterns(patterns);
    }
    patterns.insert<ContractionBiasAddRewriter>(helper_);
    // Insert multiple pattern rewriters from template instantiations by
    // activation ops.
    InsertPatterns<ContractionBiasAddActivationRewriter, OpKind::Relu,
                   OpKind::Relu6, OpKind::Elu, OpKind::LeakyRelu, OpKind::Tanh,
                   OpKind::Sigmoid>(patterns, helper_);
  }

  void populateRemapperPDLLPatterns(RewritePatternSet &patterns) {
    mkl::populateGeneratedPDLLPatterns(patterns);
  }

  FrozenRewritePatternSet final_patterns_;
  OpPropertyHelper helper_;
};

void Remapper::runOnOperation() {
  if (failed(applyPatternsAndFoldGreedily(getOperation(), final_patterns_))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> CreateRemapperPass(bool enable_onednn_patterns,
                                         bool xla_auto_clustering) {
  return std::make_unique<Remapper>(enable_onednn_patterns,
                                    xla_auto_clustering);
}

}  // namespace tfg
}  // namespace mlir
