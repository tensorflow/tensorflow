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
#include <type_traits>
#include <utility>

#include "mlir/Dialect/PDL/IR/PDL.h"                     // from @llvm-project
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"         // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                        // from @llvm-project
#include "mlir/IR/OperationSupport.h"                    // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Parser/Parser.h"                          // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/transforms/pass_detail.h"
#include "tensorflow/core/transforms/remapper/remapping_helper.h"
#include "tensorflow/core/transforms/utils/pdll/utils.h"
#include "tensorflow/core/transforms/utils/utils.h"
#include "tensorflow/core/util/util.h"

namespace mlir {
namespace tfg {
namespace mkl {
#include "tensorflow/core/transforms/remapper/pdll/MklPDLLPatterns.h.inc"
}  // namespace mkl

using namespace remapping;

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
        !dtype_attr.getValue().isa<BFloat16Type>())
      return failure();

    if (!util::NodeIsOnCpu(op)) return failure();

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

namespace {
class RewriterBase : public RewritePattern {
 public:
  RewriterBase(StringRef opName, OpPropertyHelper &helper,
               PatternBenefit benefit = PatternBenefit(1))
      : RewritePattern(opName, benefit, helper.getDialect()->getContext()),
        helper_(helper),
        dialect_(helper.getDialect()) {}
  RewriterBase(MatchAnyOpTypeTag tag, OpPropertyHelper &helper,
               PatternBenefit benefit = PatternBenefit(1))
      : RewritePattern(tag, benefit, helper.getDialect()->getContext()),
        helper_(helper),
        dialect_(helper.getDialect()) {}

 protected:
  OpPropertyHelper &helper_;
  TFGraphDialect *dialect_;
};
}  // namespace

static FailureOr<TFOp> CreateContractionBiasAddOp(PatternRewriter &rewriter,
                                                  OpPropertyHelper &helper,
                                                  Operation *contraction_op,
                                                  Operation *bias_add_op) {
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
    return rewriter.notifyMatchFailure(contraction_op,
                                       "Unsupported contraction op.");
  }

  SmallVector<Value, 4> operands;
  Value input = contraction_op->getOperand(0);
  Value filter = contraction_op->getOperand(1);
  Value bias = bias_add_op->getOperand(1);
  operands.push_back(input);
  operands.push_back(filter);
  operands.push_back(bias);
  llvm::append_range(operands, TFOp(contraction_op).getControlOperands());
  llvm::append_range(operands, TFOp(bias_add_op).getControlOperands());

  SmallVector<Location, 4> fused_locs{contraction_op->getLoc(),
                                      bias_add_op->getLoc()};

  NamedAttrList fused_attrs(contraction_op->getAttrs());
  fused_attrs.set("fused_ops", rewriter.getStrArrayAttr({"BiasAdd"}));
  fused_attrs.set("num_args", rewriter.getI32IntegerAttr(1));
  // Default values for epsilon and leakyrelu_alpha
  fused_attrs.set("epsilon", rewriter.getF32FloatAttr(0.0001));
  fused_attrs.set("leakyrelu_alpha", rewriter.getF32FloatAttr(0.2));

  OperationState state(rewriter.getFusedLoc(fused_locs), fused_op_name,
                       operands, bias_add_op->getResultTypes(), fused_attrs);

  if (auto new_op = rewriter.create(state)) {
    return TFOp(new_op);
  }

  return failure();
}

// Contraction + BiasAdd
// TODO(intel-tf): Support Contraction + {Add, AddV2} fusion in the case it has
// similar semantic of contraction + BiasAdd
class ContractionBiasAddRewriter : public RewriterBase {
 public:
  explicit ContractionBiasAddRewriter(OpPropertyHelper &helper)
      : RewriterBase("tfg.BiasAdd", helper, PatternBenefit(1)) {}

  // Constructor used by derived pattern rewritter class that may have
  // different root operation name. Currently, pattern is
  // matched from root op to its inputs.
  explicit ContractionBiasAddRewriter(StringRef op_name,
                                      OpPropertyHelper &helper,
                                      PatternBenefit benefit)
      : RewriterBase(op_name, helper, benefit) {}

  using Pattern = ContractionBiasAdd;

  bool matchPattern(Operation *op, Pattern &pattern) const {
    // Not allowing control flow on BiasAdd
    if (helper_.HasControlOperandsOrResultUsers(op)) return false;
    Operation *contraction_op = op->getOperand(0).getDefiningOp();
    if (!helper_.IsContraction(contraction_op) ||
        helper_.HasControlOperandsOrResultUsers(contraction_op) ||
        !helper_.HaveSameDataType(op, contraction_op) ||
        !helper_.HasAtMostOneUserOfResult0(contraction_op))
      return false;

    pattern.contraction = contraction_op;
    pattern.bias_add = op;
    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Pattern pattern;
    if (!matchPattern(op, pattern)) return failure();
    if (!helper_.IsDeviceCompatible(pattern)) return failure();
    FailureOr<TFOp> fused_op = CreateContractionBiasAddOp(
        rewriter, helper_, pattern.contraction, pattern.bias_add);
    if (failed(fused_op)) return failure();
    fused_op->setName(TFOp(op).nameAttr());
    rewriter.replaceOp(op, (*fused_op)->getResults());
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
    if (this->dialect_->IsNoOp(op)) return false;
    if (this->helper_.HasControlOperandsOrResultUsers(op)) return false;

    // TODO(intel-tf): Add support for more patterns.
    if constexpr (std::is_same<BasePattern, ContractionBiasAdd>::value &&
                  std::is_same<Pattern, ContractionBiasAddActivation>::value) {
      Operation *bias_add_op = op->getOperand(0).getDefiningOp();
      if (!this->dialect_->IsBiasAdd(bias_add_op) ||
          !this->helper_.HaveSameDataType(op, bias_add_op) ||
          !this->helper_.HasAtMostOneUserOfResult0(bias_add_op))
        return false;
      if (!BasePatternRewriter::matchPattern(bias_add_op, base_pattern))
        return false;
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
    if constexpr (std::is_same<BasePatternRewriter,
                               ContractionBiasAddRewriter>::value) {
      if (!this->helper_.IsDeviceCompatible(pattern)) return failure();
      Operation *&contraction_op = pattern.contraction;
      Operation *&bias_add_op = pattern.bias_add;
      Operation *&activation_op = pattern.activation;
      FailureOr<TFOp> fused_op = CreateContractionBiasAddOp(
          rewriter, this->helper_, contraction_op, bias_add_op);
      if (failed(fused_op)) return failure();
      const std::string activation_op_name =
          activation_op->getName().stripDialect().str();
      // Currently, supported activations are:
      //    _FusedMatMul: Relu, Relu6, Elu, LeakyRelu, Tanh, and Sigmoid
      //    _Fused*Conv*: Relu, Relu6, Elu and LeakyRelu
      if ((activation_op_name == "Tanh" || activation_op_name == "Sigmoid") &&
          !this->dialect_->IsMatMul(contraction_op)) {
        return failure();
      }
      (*fused_op)->setAttr("fused_ops", rewriter.getStrArrayAttr(
                                            {"BiasAdd", activation_op_name}));
      if (this->dialect_->IsLeakyRelu(activation_op))
        (*fused_op)->setAttr("leakyrelu_alpha",
                             activation_op->getAttr("alpha"));
      fused_op->setName(TFOp(op).nameAttr());
      SmallVector<Location, 4> fused_locs{(*fused_op)->getLoc(),
                                          activation_op->getLoc()};
      (*fused_op)->setLoc(rewriter.getFusedLoc(fused_locs));
      rewriter.replaceOp(op, (*fused_op)->getResults());
      return success();
    }

    return failure();
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

class Remapper : public RemapperBase<Remapper> {
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
    helper_ = std::make_shared<OpPropertyHelper>(
        context->getOrLoadDialect<TFGraphDialect>(), tensorflow::IsMKLEnabled(),
        xla_auto_clustering_);
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
    patterns.insert<ContractionBiasAddRewriter>(*helper_);
    // Insert multiple pattern rewriters from template instantiations by
    // activation ops.
    InsertPatterns<ContractionBiasAddActivationRewriter, OpKind::Relu,
                   OpKind::Relu6, OpKind::Elu, OpKind::LeakyRelu, OpKind::Tanh,
                   OpKind::Sigmoid>(patterns, *helper_);
  }

  void populateRemapperPDLLPatterns(RewritePatternSet &patterns) {
    mkl::populateGeneratedPDLLPatterns(patterns);
  }

  FrozenRewritePatternSet final_patterns_;
  std::shared_ptr<OpPropertyHelper> helper_;
};

void Remapper::runOnOperation() {
  if (failed(applyPatternsAndFoldGreedily(getOperation(), final_patterns_)))
    signalPassFailure();
}

std::unique_ptr<Pass> CreateRemapperPass(bool enable_onednn_patterns,
                                         bool xla_auto_clustering) {
  return std::make_unique<Remapper>(enable_onednn_patterns,
                                    xla_auto_clustering);
}

}  // namespace tfg
}  // namespace mlir
