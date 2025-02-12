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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/PDL/IR/PDL.h"  // from @llvm-project
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/transforms/remapper/remapping_helper.h"
#include "tensorflow/core/transforms/utils/pdll/utils.h"
#include "tensorflow/core/transforms/utils/utils.h"

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
    if (!mlir::isa<Float32Type>(dtype_attr.getValue()) &&
        !mlir::isa<BFloat16Type>(dtype_attr.getValue())) {
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
  // Setting FusedConv2D specific attrs
  if (helper.getDialect()->IsConv2D(contraction_op)) {
    TypeAttr dtype_attr = contraction_op->getAttrOfType<TypeAttr>("T");
    state->attributes.set("TArgs", builder.getArrayAttr({dtype_attr}));
    state->attributes.set("num_host_args", builder.getI32IntegerAttr(0));
  }
  // Default values for epsilon and leakyrelu_alpha
  state->attributes.set("epsilon", builder.getF32FloatAttr(0.0001));
  state->attributes.set("leakyrelu_alpha", builder.getF32FloatAttr(0.2));
  return state;
}

// AsString + StringToHashBucketFast -> _TensorToHashBucketFast
class MatchStringToHashBucket : public RemapperPatternBase {
 public:
  explicit MatchStringToHashBucket(OpPropertyHelper &helper)
      : RemapperPatternBase("tfg.StringToHashBucketFast", helper,
                            PatternBenefit(/*benefit=*/1)) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Not allowing control flow on op
    if (helper_.HasControlOperandsOrResultUsers(op)) return failure();

    if (op->getOperands().size() < 1) return failure();

    TFOp stringtohash_wrapper(op);

    // AsString op
    Operation *as_string_op = op->getOperand(0).getDefiningOp();
    if (!as_string_op) return failure();

    // Input to StringToHashBucket should be the AsString op
    if (!this->helper_.getDialect()->IsAsString(as_string_op)) return failure();

    if (helper_.HasControlOperandsOrResultUsers(as_string_op) ||
        !helper_.HasAtMostOneUserOfResult0(as_string_op)) {
      return failure();
    }

    // DataType of AsString must be int8/16/32/64
    TypeAttr dtype_attr = as_string_op->getAttrOfType<TypeAttr>("T");
    if (!dtype_attr) return failure();
    Type dtype = dtype_attr.getValue();
    if (!mlir::isa<IntegerType>(dtype)) return failure();

    // width/fill attributes must be default values
    auto width_attr = as_string_op->getAttrOfType<IntegerAttr>("width");
    if (!width_attr) return failure();
    if (width_attr.getInt() != -1) return failure();
    auto fill_attr = as_string_op->getAttrOfType<StringAttr>("fill");
    if (!fill_attr) return failure();
    if (fill_attr.getValue() != "") return failure();

    // An input to the AsString must exist to determine the device.
    if (as_string_op->getOperands().size() < 1) return failure();

    // AsString op's input
    Value input_value = as_string_op->getOperand(0);
    Operation *input_op = input_value.getDefiningOp();

    SmallVector<Value> operands;
    operands.push_back(input_value);
    // Control operands come after regular operands.
    llvm::append_range(operands, stringtohash_wrapper.getControlOperands());

    NamedAttrList fused_attrs(op->getAttrs());
    fused_attrs.set("T", dtype_attr);

    Operation *fused_op = rewriter.create(
        op->getLoc(), rewriter.getStringAttr("tfg._TensorToHashBucketFast"),
        operands, op->getResultTypes(), fused_attrs);
    TFOp(fused_op).setRequestedDevice(TFOp(input_op).deviceAttr());
    rewriter.replaceOp(op, fused_op->getResults());
    return success();
  }
};

// Convert Softplus+Tanh+Mul to Mish
// Mul(x, Tanh(Softplus(x))) --> _MklFusedMish
class MatchSoftplusTanhMul : public RemapperPatternBase {
 public:
  explicit MatchSoftplusTanhMul(OpPropertyHelper &helper)
      : RemapperPatternBase("tfg.Mul", helper, PatternBenefit(1)) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Fusion only available for CPU
    if (!util::OpHasDevice(op, tensorflow::DEVICE_CPU)) return failure();

    // Not allowing control flow on op
    if (helper_.HasControlOperandsOrResultUsers(op)) return failure();

    // Fusion only available for float32 and bfloat16 data types
    auto attr = op->getAttrOfType<TypeAttr>("T");
    if (!attr) return failure();
    Type dtype = attr.getValue();
    if (!mlir::isa<Float32Type, BFloat16Type>(dtype)) return failure();

    TFOp mul_wrapper(op);

    // Tanh op
    Value tanh_value = op->getOperand(0);
    // Input
    Value x_value = op->getOperand(1);

    // The Mul op is commutative and the inputs may be swapped.
    auto CheckTanhOperand = [&](Value tanh_value) {
      if (!tanh_value) return false;
      Operation *op = tanh_value.getDefiningOp();
      return op && this->helper_.getDialect()->IsTanh(op);
    };

    if (!CheckTanhOperand(tanh_value)) {
      std::swap(tanh_value, x_value);
      if (!CheckTanhOperand(tanh_value)) return failure();
    }

    Operation *tanh_op = tanh_value.getDefiningOp();

    // Softplus op
    Value softplus_value = tanh_op->getOperand(0);
    Operation *softplus_op = softplus_value.getDefiningOp();

    if (!(this->helper_.getDialect()->IsSoftplus(op)) &&
        !(softplus_op->getOperand(0) == x_value)) {
      return failure();
    }

    if (!helper_.HasAtMostOneUserOfResult0(tanh_op) ||
        !helper_.HasAtMostOneUserOfResult0(softplus_op)) {
      return failure();
    }

    // TODO(intel-tf): Allow valid control dependencies
    // Not allowing control flow on Tanh or Softplus
    if (helper_.HasControlOperandsOrResultUsers(tanh_op) ||
        helper_.HasControlOperandsOrResultUsers(softplus_op)) {
      return failure();
    }

    SmallVector<Value> operands;
    // Set up non-control operand.
    operands.push_back(x_value);
    // Control operands come after regular operands.
    llvm::append_range(operands, mul_wrapper.getControlOperands());

    Operation *new_op = rewriter.create(
        op->getLoc(), rewriter.getStringAttr("tfg._MklFusedMish"), operands,
        op->getResultTypes(), op->getAttrs());
    rewriter.replaceOp(op, new_op->getResults());

    return success();
  }
};

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

// NOTE(ezhulenev): See `BatchnormSpatialPersistentEnabled` documentation in the
// `tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc` for details.
bool BatchnormSpatialPersistentEnabled() {
#if CUDNN_VERSION >= 7402
  static bool is_enabled = [] {
    bool is_enabled = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
        "TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT",
        /*default_val=*/false, &is_enabled));
    return is_enabled;
  }();
  return is_enabled;
#else
  return false;
#endif
}

// FusedBatchNorm[$is_training] + ... -> _FusedBatchNormEx[$is_training]
//   (1) FusedBatchNorm + <Activation>
//   (2) FusedBatchNorm + SideInput + <Activation>
// only supported activation is Relu
class FusedBatchNormExRewriter : public RemapperPatternBase {
 public:
  explicit FusedBatchNormExRewriter(OpPropertyHelper &helper)
      : RemapperPatternBase("tfg.Relu", helper, PatternBenefit(1)) {}

  // Constructor used by derived pattern rewritter class that may have
  // different root operation name. Currently, pattern is
  // matched from root op to its inputs.
  explicit FusedBatchNormExRewriter(StringRef op_name, OpPropertyHelper &helper,
                                    PatternBenefit benefit)
      : RemapperPatternBase(op_name, helper, benefit) {}

  using Pattern = FusedBatchNormEx;

  bool is_valid_batch_norm(Operation *fused_batch_norm_op) const {
    TFOp fusedbatchnormop_wrapper(fused_batch_norm_op);
    if (!this->helper_.getDialect()->IsFusedBatchNorm(
            fusedbatchnormop_wrapper)) {
      return false;
    }
    // We fuse FusedBatchNorm on GPU or oneDNN CPU.
    if (!this->helper_.isOneDNNEnabled() &&
        !util::OpHasDevice(fused_batch_norm_op, tensorflow::DEVICE_GPU)) {
      return false;
    }

    TypeAttr attr = fused_batch_norm_op->getAttrOfType<TypeAttr>("T");
    if (!attr) return false;
    Type dtype_T = attr.getValue();

    if (util::OpHasDevice(fused_batch_norm_op, tensorflow::DEVICE_GPU)) {
      // GPU supports float and half.
      // Put this condition before check `isOneDNNEnabled()` because this node
      // should be processed when it's on GPU and oneDNN CPU is enabled.
      if (!mlir::isa<Float32Type, Float16Type>(dtype_T)) return false;
    } else {
      // Bfloat16 is available only with oneDNN.
      // Half is not available with oneDNN.
      if (this->helper_.isOneDNNEnabled() &&
          !mlir::isa<Float32Type, BFloat16Type>(dtype_T)) {
        return false;
      }
    }

    // Get the FusedBatchNorm training mode.
    auto training_attr =
        fused_batch_norm_op->getAttrOfType<BoolAttr>("is_training");
    if (!training_attr) return false;
    bool is_training = training_attr.getValue();

    auto data_format_attr =
        fused_batch_norm_op->getAttrOfType<StringAttr>("data_format");
    if (!data_format_attr) return false;
    StringRef data_format = data_format_attr.getValue();

    if (data_format != "NHWC" && data_format != "NCHW") return false;

    // In training mode we rely on cuDNN for computing FusedBatchNorm with side
    // inputs and activation, and it has its own limitations. In inference mode
    // we have a custom CUDA kernel that doesn't not have these constraints.
    if (is_training &&
        util::OpHasDevice(fused_batch_norm_op, tensorflow::DEVICE_GPU)) {
      // cuDNN only supports NHWC data layout.
      if (data_format != "NHWC") return false;

      // Data type must be Float16.
      if (!mlir::isa<Float16Type>(dtype_T)) return false;

      // Channel dimension must be a multiple of 4.
      auto fbn_input0_shape =
          mlir::cast<ShapedType>(fused_batch_norm_op->getOperand(0).getType());
      auto fbn_input0_shape_dims = fbn_input0_shape.getShape();

      const bool valid_channel_dim = (fbn_input0_shape.getRank() == 4) &&
                                     (fbn_input0_shape_dims[3] % 4 == 0);

      if (!valid_channel_dim) return false;

      // cuDNN must support CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode.
      if (!BatchnormSpatialPersistentEnabled()) return false;
    }

    // FusedBatchNormV2 and V3 have an extra type parameter.
    if (fused_batch_norm_op->getName().getStringRef() != "tfg.FusedBatchNorm") {
      auto attr = fused_batch_norm_op->getAttrOfType<TypeAttr>("U");
      if (attr && !mlir::isa<Float32Type>(attr.getValue())) {
        return false;
      }
    }

    // Check that only one node consumes the 0-th output of a FusedBatchNorm.
    if (this->helper_.HasControlOperandsOrResultUsers(fused_batch_norm_op) ||
        !this->helper_.HasAtMostOneUserOfResult0(fused_batch_norm_op)) {
      return false;
    }

    return true;
  }

  bool matchPattern(Operation *op, Pattern &pattern) const {
    TFOp activation_tfg_wrapper(op);
    // Not allowing control flow on Relu
    if (helper_.HasControlOperandsOrResultUsers(op)) return false;
    if (activation_tfg_wrapper.getNonControlOperands().empty()) return false;

    Operation *activation_input_op = op->getOperand(0).getDefiningOp();
    if (activation_input_op == nullptr) return false;
    if (is_valid_batch_norm(activation_input_op)) {
      pattern.fused_batch_norm = activation_input_op;
      pattern.activation = op;
      pattern.side_input = nullptr;
      return true;
    }

    // Input to a Relu can be an Add node with FusedBatchNorm as one of the
    // inputs
    if (this->helper_.getDialect()->IsAdd(activation_input_op)) {
      // Currently no CPU implementation for "FusedBatchNorm + SideInput +
      // <Activation>"
      if (this->helper_.isOneDNNEnabled() &&
          !util::OpHasDevice(op, tensorflow::DEVICE_GPU)) {
        return false;
      }

      // Check that only Relu node consumes the output of an Add node.
      if (helper_.HasControlOperandsOrResultUsers(activation_input_op) ||
          !helper_.HasAtMostOneUserOfResult0(activation_input_op)) {
        return false;
      }

      if (activation_input_op->getOperands().size() < 2 &&
          TFOp(activation_input_op).getNonControlOperands().size() < 2) {
        return false;
      }

      // Add node supports broadcasting, FusedBatchNormEx does not.
      // Check for symbolic shape equivalence
      auto add_input0_op = activation_input_op->getOperand(0).getDefiningOp();
      auto add_input1_op = activation_input_op->getOperand(1).getDefiningOp();
      if (add_input0_op == nullptr || add_input1_op == nullptr) return false;
      auto add_input0_shape =
          mlir::cast<ShapedType>(activation_input_op->getOperand(0).getType());
      auto add_input1_shape =
          mlir::cast<ShapedType>(activation_input_op->getOperand(1).getType());
      if (add_input0_shape.getShape() != add_input1_shape.getShape()) {
        return false;
      }

      if (is_valid_batch_norm(add_input0_op)) {
        pattern.fused_batch_norm = add_input0_op;
        pattern.activation = op;
        pattern.side_input = activation_input_op->getOperand(1);
        return true;
      }

      if (is_valid_batch_norm(add_input1_op)) {
        pattern.fused_batch_norm = add_input1_op;
        pattern.activation = op;
        pattern.side_input = activation_input_op->getOperand(0);
        return true;
      }
    }

    return false;
  }

  LogicalResult createFusedBatchNormExOpState(OpBuilder &builder,
                                              FusedBatchNormEx *pattern,
                                              OperationState &state) const {
    Operation *fused_batch_norm = pattern->fused_batch_norm;
    Operation *activation = pattern->activation;
    Value side_input = pattern->side_input;

    state.addOperands(fused_batch_norm->getOperands());
    if (side_input) {
      state.operands.push_back(side_input);
    }
    state.addOperands(TFOp(fused_batch_norm).getControlOperands());
    state.addTypes(fused_batch_norm->getResultTypes());
    state.attributes = fused_batch_norm->getAttrs();
    state.attributes.set(
        "activation_mode",
        builder.getStringAttr(activation->getName().stripDialect()));
    if (side_input) {
      state.attributes.set("num_side_inputs", builder.getI32IntegerAttr(1));
    } else {
      state.attributes.set("num_side_inputs", builder.getI32IntegerAttr(0));
    }
    return success();
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Pattern pattern;

    if (!matchPattern(op, pattern)) return failure();

    OperationState state(op->getLoc(), "tfg._FusedBatchNormEx");
    LogicalResult create_op_state =
        createFusedBatchNormExOpState(rewriter, &pattern, state);
    if (!succeeded(create_op_state)) return failure();

    Operation *fused_op = rewriter.create(state);

    auto fused_batch_norm_op_name = TFOp(pattern.fused_batch_norm).nameAttr();
    TFOp(fused_op).setName(fused_batch_norm_op_name);

    OperationState identity_op_state(UnknownLoc::get(rewriter.getContext()),
                                     "tfg.Identity");
    identity_op_state.addAttribute("T", op->getAttr("T"));
    identity_op_state.addOperands(fused_op->getResult(0));
    identity_op_state.addTypes(op->getResultTypes());
    Operation *identity_op = rewriter.create(identity_op_state);
    TFOp(identity_op).setName(TFOp(op).nameAttr());
    if (!TFOp(op).device().empty())
      TFOp(identity_op).setRequestedDevice(TFOp(op).deviceAttr());

    rewriter.replaceOp(op, identity_op->getResults());
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
      patterns.insert<MatchSoftplusTanhMul>(helper_);
      // TODO(chiahungduan): Currently, the only pattern implemented in PDLL is
      // the same one as `MatchMulSigmoid`. Remove the one of them when there's
      // a decision that which one is preferred.
      populateRemapperPDLLPatterns(patterns);
    }
    patterns.insert<ContractionBiasAddRewriter>(helper_);
    patterns.insert<MatchStringToHashBucket>(helper_);
    // Insert multiple pattern rewriters from template instantiations by
    // activation ops.
    InsertPatterns<ContractionBiasAddActivationRewriter, OpKind::Relu,
                   OpKind::Relu6, OpKind::Elu, OpKind::LeakyRelu, OpKind::Tanh,
                   OpKind::Sigmoid>(patterns, helper_);
    patterns.insert<FusedBatchNormExRewriter>(helper_);
  }

  void populateRemapperPDLLPatterns(RewritePatternSet &patterns) {
    mkl::populateGeneratedPDLLPatterns(patterns);
  }

  FrozenRewritePatternSet final_patterns_;
  OpPropertyHelper helper_;
};

void Remapper::runOnOperation() {
  if (failed(applyPatternsGreedily(getOperation(), final_patterns_))) {
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
