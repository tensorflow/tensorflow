/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass takes operations in TensorFlowLite dialect and
// optimizes them to resulting operations in TensorFlowLite dialect.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Optimize Pass.
namespace {
constexpr char kRelu[] = "RELU";
constexpr char kRelu6[] = "RELU6";
constexpr char kRelu1[] = "RELU_N1_TO_1";

bool L2NormalizeReduceAxis(Value sq_op, DenseElementsAttr axis) {
  if (sq_op.getType().cast<ShapedType>().getRank() - 1 ==
          *axis.getValues<int>().begin() ||
      *axis.getValues<int>().begin() == -1) {
    return true;
  }
  if (sq_op.getType().cast<ShapedType>().getRank() != axis.getNumElements()) {
    return false;
  }
  auto shape = sq_op.getType().cast<ShapedType>();
  SmallVector<int, 4> elems{axis.getValues<int>().begin(),
                            axis.getValues<int>().end()};
  for (int i = 0; i < shape.getRank(); ++i) {
    if (i != elems[i]) return false;
  }
  return true;
}

using ::llvm::cast;

// Optimize TFLite operations in functions.
struct Optimize : public PassWrapper<Optimize, FunctionPass> {
  void runOnFunction() override;
};

// Returns whether the given type `a` is broadcast-compatible with `b`.
bool IsBroadcastableElementsAttrAndType(Type a, Type b) {
  return OpTrait::util::getBroadcastedType(a, b) != Type();
}

// Returns whether the resultant type of any broadcastable operation with
// operands `a` and `b` matches `expected_output`. Returns false if `a` is not
// broadcast-compatible with `b`.
bool OperandsBroadcastToOutputType(Type a, Type b, Type expected_output) {
  Type output_element_type =
      expected_output.cast<ShapedType>().getElementType();
  Type broadcasted_type =
      OpTrait::util::getBroadcastedType(a, b, output_element_type);
  return broadcasted_type != Type() && broadcasted_type == expected_output;
}

// Returns whether if `type1` dimensions are the same as the ending dimensions
// of `type2`. This is more restricted than broadcastable.
bool IsTailOfShape(Type type1, Type type2) {
  auto tail_type = type1.dyn_cast<ShapedType>();
  auto full_type = type2.dyn_cast<ShapedType>();
  if (!tail_type || !full_type || tail_type.getRank() > full_type.getRank())
    return false;
  auto i1 = tail_type.getShape().rbegin(), e1 = tail_type.getShape().rend();
  auto i2 = full_type.getShape().rbegin();
  return std::equal(i1, e1, i2);
}

bool CanFuseConvOrDepthwiseConvShapes(const ArrayRef<int64_t> filter_shape,
                                      const ArrayRef<int64_t> elements_shape,
                                      bool is_depthwise) {
  // Make sure the val tensor has shape where all dimensions are 1 except
  // last one.
  // Also, val tensor must be of rank 1 or 4 or 0 (scalar).
  const auto elements_rank = elements_shape.size();
  for (int i = 0; i < static_cast<int>(elements_shape.size()) - 1; ++i) {
    if (elements_shape[i] != 1) return false;
  }
  if (elements_rank != 1 && elements_rank != 0 && elements_rank != 4) {
    return false;
  }
  auto elements_depth = elements_shape.empty() ? 1 : elements_shape.back();
  // In TFLite Conv2D uses OHWI format for filter, and 1HWO for Depthwise Conv.
  // For conv:
  // Check if last dimension in filter equals the first dimension
  // For depthwise conv:
  // Check if the first in filter dimension equals the first dimension.
  if (filter_shape.empty() ||
      (is_depthwise ? filter_shape.back() != elements_depth
                    : filter_shape[0] != elements_depth))
    return false;
  return true;
}

bool CanFuseConvOrDepthwiseConv(Value filter, Attribute val,
                                bool is_depthwise) {
  const auto elements = val.dyn_cast<DenseElementsAttr>();
  if (!elements) {
    return false;
  }
  const auto elements_shape = elements.getType().getShape();
  const auto filter_shape = filter.getType().cast<ShapedType>().getShape();
  return CanFuseConvOrDepthwiseConvShapes(filter_shape, elements_shape,
                                          is_depthwise);
}

bool CanFuseConvOrDepthwiseConv(Attribute filter, Attribute val,
                                bool is_depthwise) {
  if (const auto elements = val.dyn_cast<DenseElementsAttr>()) {
    if (const auto filter_elements = filter.dyn_cast<DenseElementsAttr>()) {
      return CanFuseConvOrDepthwiseConvShapes(
          filter_elements.getType().getShape(), elements.getType().getShape(),
          is_depthwise);
    }
  }
  return false;
}

// Expand Attribute 'a' to 4D with all 1s except 1 dimension.
// Which dimension depends on 'is_depthwise' is true or false.
ElementsAttr ExpandTo4DForConvImpl(Attribute a, bool is_depthwise) {
  auto elements = a.dyn_cast<DenseElementsAttr>();
  auto shape = elements.getType().getShape();
  if (shape.size() == 4) {
    return elements;
  }
  std::vector<int64_t> shape_data = {1, 1, 1, 1};
  if (shape.size() == 1 || shape.empty()) {
    if (is_depthwise)
      shape_data[3] = shape.empty() ? 1 : shape[0];
    else
      shape_data[0] = shape.empty() ? 1 : shape[0];
  }
  auto new_shape =
      RankedTensorType::get(shape_data, elements.getType().getElementType());
  return elements.reshape(new_shape);
}

ElementsAttr ExpandTo4DForConv(Attribute a) {
  return ExpandTo4DForConvImpl(a, false);
}

ElementsAttr ExpandTo4DForDepthwiseConv(Attribute a) {
  return ExpandTo4DForConvImpl(a, true);
}

TypeAttr RescaleQtype(Type input, Attribute factor) {
  return quant::RescaleQuantizedType(input, factor);
}

// Returns shape of a ranked tensor.
// Precondition: output_val's is ranked tensor.
DenseElementsAttr GetShape(Value output_val) {
  auto output_type = output_val.getType().cast<RankedTensorType>();
  auto shape_vector = output_type.getShape();
  std::vector<int32_t> shape(shape_vector.size());
  for (int i = 0; i < shape_vector.size(); ++i) {
    shape[i] = shape_vector[i];
  }
  return mlir::DenseElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int>(shape.size())},
          mlir::IntegerType::get(32, output_val.getContext())),
      llvm::makeArrayRef(shape));
}

static Type GetShapeStrippedType(TypeAttr type_attr) {
  auto type = type_attr.getValue();
  auto shaped_type = type.dyn_cast<ShapedType>();
  if (shaped_type) {
    return shaped_type.getElementType();
  } else {
    return type;
  }
}

bool NotFromQuantOpDifferentQuant(Value val, TypeAttr qtype_attr) {
  auto val_defn_op = val.getDefiningOp();
  TFL::QuantizeOp q_op = llvm::dyn_cast_or_null<TFL::QuantizeOp>(val_defn_op);
  if (!q_op) return true;

  // Ignore shape details - weŕe really only trying to
  // check if quantization is the same.
  auto stripped_src_qtype = GetShapeStrippedType(q_op.qtypeAttr());
  auto stripped_qtype = GetShapeStrippedType(qtype_attr);
  return stripped_src_qtype == stripped_qtype;
}

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize.inc"

// Fuse Add with proceeding FullyConnected.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseFullyConnectedAndAdd : public OpRewritePattern<TFL::AddOp> {
  using OpRewritePattern<TFL::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::AddOp add_op,
                                PatternRewriter &rewriter) const override {
    // Match Add.
    DenseElementsAttr added_value;
    Value constant_val = add_op.rhs();
    if (!matchPattern(constant_val, m_Constant(&added_value))) return failure();

    // Match Fully Connected.
    auto fc_op =
        dyn_cast_or_null<TFL::FullyConnectedOp>(add_op.lhs().getDefiningOp());
    if (!fc_op) return failure();

    // Check if the constant RHS is either 0D (scalar), or a 1D with
    // `{num_channels}` shape.
    auto constant_val_type = constant_val.getType().cast<TensorType>();

    // In TFLite FullyConnect definition, bias must be a 1D tensor where
    // the number of elements is equal to the number of channels.
    // If it's not 1D or 0D (which can be broadcasted to 1D), reject the
    // matching.
    bool is_scalar_rhs = false;
    if (constant_val_type.getRank() == 0) {
      is_scalar_rhs = true;
    } else if (constant_val_type.getRank() != 1) {
      return failure();
    }

    Value filter = fc_op.filter();
    Value bias = fc_op.bias();
    ElementsAttr bias_value;
    const bool is_none_bias = bias.getType().isa<NoneType>();
    if (fc_op.fused_activation_function() != "NONE") return failure();

    if (!is_none_bias && !matchPattern(bias, m_Constant(&bias_value)))
      return failure();

    // Rewrite
    Location loc = fc_op.getLoc();

    if (is_none_bias) {
      if (is_scalar_rhs) {
        // If the `constant_val` is scalar, we must the shape of filter
        // to properly broadcast the scalar to `{num_channels}` shape.

        // Get the number of channels if possible.
        auto filter_type = filter.getType().cast<ShapedType>();
        // Filter must be a `2D` tensor with `{num_channels, num_features}`
        // shape. The following check is rejecting unknown rank (-1).
        if (filter_type.getRank() != 2) {
          return failure();
        }
        int num_channels = filter_type.getShape()[0];

        // Create a zero tensor with shape {num_channels}, and the type need to
        // be the same as constant_val.
        // This is a way to gracefully handle scalar tensor. The Add will always
        // be constant-folded away regardless if `constant_val` is a scalar or
        // not.
        RankedTensorType type = RankedTensorType::get(
            {num_channels}, constant_val_type.getElementType());
        auto attr = rewriter.getZeroAttr(type);
        bias = rewriter.create<ConstantOp>(loc, type, attr);
        auto none_af = rewriter.getStringAttr("NONE");
        bias =
            rewriter.create<AddOp>(loc, bias, constant_val, none_af).output();
      } else {
        // If there no pre-existing bias and the `constant_val` is 1D, simply
        // use `constant_val` as bias.
        bias = constant_val;
      }
    } else {
      auto none_af = rewriter.getStringAttr("NONE");
      bias = rewriter.create<AddOp>(loc, bias, constant_val, none_af).output();
    }

    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        add_op, add_op.getType(),
        /*input=*/fc_op.input(),
        /*filter=*/filter,
        /*bias=*/bias,
        /*fused_activation_function=*/
        rewriter.getStringAttr(add_op.fused_activation_function()),
        /*weights_format=*/rewriter.getStringAttr(fc_op.weights_format()),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.keep_num_dims()));

    return success();
  }
};

// TODO(b/136285429): Move to tablegen when variadic is supported.
template <typename ReluXOp, char const *Act>
struct FuseFullyConnectedAndReluX : public OpRewritePattern<ReluXOp> {
  using OpRewritePattern<ReluXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluXOp relu_op,
                                PatternRewriter &rewriter) const override {
    Operation *input = relu_op.getOperand().getDefiningOp();
    if (!isa_and_nonnull<FullyConnectedOp>(input)) return failure();
    auto fully_connected_op = cast<FullyConnectedOp>(input);
    if (fully_connected_op.fused_activation_function() != "NONE")
      return failure();

    auto new_activation_func = rewriter.getStringAttr(Act);
    auto new_weights_format =
        rewriter.getStringAttr(fully_connected_op.weights_format());
    auto new_keep_num_dims =
        rewriter.getBoolAttr(fully_connected_op.keep_num_dims());
    rewriter.replaceOpWithNewOp<FullyConnectedOp>(
        relu_op, relu_op.getType(), fully_connected_op.input(),
        fully_connected_op.filter(), fully_connected_op.bias(),
        new_activation_func, new_weights_format, new_keep_num_dims);

    return success();
  }
};

// Fuse Mul with proceeding FullyConnected.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseFullyConnectedAndMul : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mul_op,
                                PatternRewriter &rewriter) const override {
    // Mul.
    DenseElementsAttr cst;
    Value constant_val = mul_op.rhs();
    if (!matchPattern(constant_val, m_Constant(&cst))) return failure();

    // Fully Connected.
    auto fc_op =
        dyn_cast_or_null<TFL::FullyConnectedOp>(mul_op.lhs().getDefiningOp());
    if (!fc_op) return failure();
    Value filter = fc_op.filter();
    Value bias = fc_op.bias();
    ElementsAttr cst_tmp;
    if (!matchPattern(filter, m_Constant(&cst_tmp))) return failure();
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&cst_tmp)))
      return failure();
    if (fc_op.fused_activation_function() != "NONE") return failure();

    // Broadcast the constant operand of Mul if it isn't compatible to the
    // filter input. We only support broadcasting the operand along the depth
    // dimension, when the operand's depth is 1.
    Value new_const_val = constant_val;
    if (!IsBroadcastableElementsAttrAndType(cst.getType(), filter.getType())) {
      auto original_shape = cst.getType().getShape();
      llvm::SmallVector<int64_t, 4> normalized_shape(original_shape.begin(),
                                                     original_shape.end());
      normalized_shape.push_back(1);
      auto new_cst = cst.reshape(RankedTensorType::get(
          normalized_shape, cst.getType().getElementType()));
      Type new_type = new_cst.getType();
      if (!IsBroadcastableElementsAttrAndType(new_type, filter.getType())) {
        return failure();
      }
      auto new_op =
          rewriter.create<ConstantOp>(mul_op.getLoc(), new_type, new_cst);
      new_const_val = new_op.getResult();
    }

    // Rewrite. Since the folder of TFL::MulOp couldn't broadcast the operands,
    // TF::MulOp is used to fold the constant.
    // TODO(b/139192933): switch to the TFL constant folding
    Location loc = fc_op.getLoc();
    auto new_filter =
        rewriter.create<TF::MulOp>(loc, filter, new_const_val).z();
    // If bias isn't None, it needs to be multiplied as well.
    if (!bias.getType().isa<NoneType>()) {
      bias = rewriter.create<TF::MulOp>(loc, bias, constant_val).z();
    }

    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        mul_op, mul_op.getType(),
        /*input=*/fc_op.input(),
        /*filter=*/new_filter,
        /*bias=*/bias,
        /*fused_activation_function=*/
        rewriter.getStringAttr(mul_op.fused_activation_function()),
        /*weights_format=*/rewriter.getStringAttr(fc_op.weights_format()),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.keep_num_dims()));

    return success();
  }
};

// Fuse Mul with proceeding Affine ops. This is an C++ implementation of the
// following table gen implementation, which doesn't derived the result type of
// the TFL_DequantizeOp.
// def : Pat<(TFL_MulOp (TFL_Conv2DOp:$conv_output $input,
//                          (TFL_DequantizeOp (TFL_QuantizeOp
//                              (ConstantOp F32ElementsAttr:$filter), $qtype)),
//                          (ConstantOp F32ElementsAttr:$bias),
//                          $h_factor, $w_factor, TFL_AF_None,
//                          $padding, $stride_h, $stride_w),
//                      (ConstantOp F32ElementsAttr:$value), $act_fn),
//           (TFL_Conv2DOp $input,
//                      (TFL_DequantizeOp (TFL_QuantizeOp
//                          (TFL_MulOp (ConstantOp $filter),
//                                     (ConstantOp (ExpandTo4DForConv $value)),
//                                      TFL_AF_None),
//                          (RescaleQtype $qtype, $value))),
//                      (TFL_MulOp (ConstantOp $bias), (ConstantOp $value),
//                          TFL_AF_None),
//                      $h_factor, $w_factor, $act_fn,
//                      $padding, $stride_h, $stride_w),
//         [(CanFuseConvOrDepthwiseConv<"false"> $filter, $value),
//          (HasOneUse $conv_output),
//          (IsPerAxisQuantization $qtype), // per-axis quantization
//         ]>;
template <typename AffineOpType>
struct FuseAffinOpAndMulWithQDQs : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mul_op,
                                PatternRewriter &rewriter) const override {
    // Mul. Required 1-D rhs for batch normalization.
    DenseElementsAttr gamma_cst;
    Value gamma = mul_op.rhs();
    if (!matchPattern(gamma, m_Constant(&gamma_cst))) return failure();
    if (gamma_cst.getType().getRank() != 1) return failure();

    // Affine op
    Operation *mul_op_lhs = mul_op.lhs().getDefiningOp();
    auto fc_op = dyn_cast_or_null<AffineOpType>(mul_op_lhs);
    if (!fc_op) return failure();
    Value filter = fc_op.filter();
    Value bias = fc_op.bias();

    // QDQs
    auto dq_op = dyn_cast_or_null<TFL::DequantizeOp>(filter.getDefiningOp());
    if (!dq_op) return failure();
    auto q_op =
        dyn_cast_or_null<TFL::QuantizeOp>(dq_op.input().getDefiningOp());
    if (!q_op) return failure();
    filter = q_op.input();

    // weight constant
    ElementsAttr cst_tmp;
    if (!matchPattern(filter, m_Constant(&cst_tmp))) return failure();
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&cst_tmp)))
      return failure();
    if (fc_op.fused_activation_function() != "NONE") return failure();

    // Broadcast the constant operand of Mul if it isn't compatible to the
    // filter input. We only support broadcasting the operand along the depth
    // dimension, when the operand's depth is 1.
    rewriter.setInsertionPoint(q_op);
    Location loc = fc_op.getLoc();
    Value broadcasted_gamma;
    if (isa<TFL::Conv2DOp>(mul_op_lhs)) {
      auto mul_rhs = ExpandTo4DForConv(gamma_cst);
      broadcasted_gamma = rewriter.create<ConstOp>(loc, mul_rhs);
    } else if (isa<TFL::DepthwiseConv2DOp>(mul_op_lhs)) {
      auto mul_rhs = ExpandTo4DForDepthwiseConv(gamma_cst);
      broadcasted_gamma = rewriter.create<ConstOp>(loc, mul_rhs);
    } else {
      return failure();
    }

    // Rewrite filter constant. Since the folder of TFL::MulOp couldn't
    // broadcast the operands, TF::MulOp is used to fold the constant.
    auto new_filter =
        rewriter.create<TF::MulOp>(loc, filter, broadcasted_gamma).z();
    // Update the scale in the quantize op.
    auto new_qtype = RescaleQtype(q_op.qtype(), gamma_cst);
    if (!new_qtype) return failure();
    rewriter.replaceOpWithNewOp<TFL::QuantizeOp>(q_op, new_qtype.getValue(),
                                                 new_filter, new_qtype);

    // If bias isn't None, it needs to be multiplied as well.
    if (!bias.getType().isa<NoneType>()) {
      rewriter.setInsertionPoint(fc_op);
      auto new_bias = rewriter.create<TF::MulOp>(loc, bias, gamma);
      fc_op.getOperation()->replaceUsesOfWith(bias, new_bias);
    }

    // Remove the tailing mul op.
    mul_op.replaceAllUsesWith(fc_op.getResult());
    return success();
  }
};

using FuseConv2DAndMulWithQDQs = FuseAffinOpAndMulWithQDQs<TFL::Conv2DOp>;
using FuseDepthwiseConv2DAndMulWithQDQs =
    FuseAffinOpAndMulWithQDQs<TFL::DepthwiseConv2DOp>;

// Fuse Binary Op with following Affine operation.
template <typename AffineOpType>
struct FuseBinaryOpToFollowingAffineOp : public OpRewritePattern<AffineOpType> {
  using OpRewritePattern<AffineOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineOpType fc_op,
                                PatternRewriter &rewriter) const override {
    // Binary op.
    Operation *binary_op = fc_op.input().getDefiningOp();
    if (!binary_op || binary_op->getNumOperands() != 2) return failure();
    // We only handle the cases the RHS is a scalar.
    // TODO(fengliuai): Currently the canonicalizer pass couldn't guarantee that
    // the constant operands are on the RHS, we need to consider LHS constant
    // operand if necessary.
    DenseFPElementsAttr cst;
    if (!matchPattern(binary_op->getOperand(1), m_Constant(&cst)))
      return failure();
    if (cst.getNumElements() != 1) return failure();
    APFloat cst_value = *cst.float_value_begin();

    // Affine op.
    Value filter = fc_op.filter();
    Value bias = fc_op.bias();
    DenseFPElementsAttr filter_cst, bias_cst;
    if (!matchPattern(filter, m_Constant(&filter_cst))) {
      // The filter maybe quantized, then we should set it to the real constant.
      auto dq = llvm::dyn_cast_or_null<DequantizeOp>(filter.getDefiningOp());
      if (!dq) return failure();
      auto q = llvm::dyn_cast_or_null<QuantizeOp>(dq.input().getDefiningOp());
      if (!q || !matchPattern(q.input(), m_Constant(&filter_cst))) {
        return failure();
      }
      filter = q.input();
    }
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&bias_cst)))
      return failure();
    ShapedType filter_type = filter_cst.getType();

    if (llvm::isa<AddOp>(binary_op) || llvm::isa<SubOp>(binary_op)) {
      auto padding = fc_op.template getAttrOfType<StringAttr>("padding");
      if (padding && padding.getValue() != "VALID") return failure();

      // The fusion of add/sub is actually applying the following
      // transformation:
      // w * (x + c) + b => w * x + (w * c + b)
      // so we have to update the bias.
      if (llvm::isa<SubOp>(binary_op)) cst_value.changeSign();

      auto bias_and_slice =
          GetBiasDimAndSliceSize(filter_type.getShape(), fc_op);
      int64_t bias_size = bias_and_slice.first;
      int64_t slice_size = bias_and_slice.second;
      ShapedType new_bias_type =
          RankedTensorType::get({bias_size}, filter_type.getElementType());

      // The new bias should be a 1-D tensor with length equals to the bias
      // dimension of the weight.
      SmallVector<APFloat, 4> new_bias_values;
      if (bias.getType().isa<NoneType>()) {  // none bias, a list of zeros
        new_bias_values.resize(bias_size, APFloat(0.0));
      } else if (bias_cst.getNumElements() == 1) {  // scalar bias, broadcast it
        new_bias_values.resize(bias_size, *bias_cst.float_value_begin());
      } else if (bias_cst.getNumElements() == bias_size) {  // 1-d bias, copy it
        new_bias_values.insert(new_bias_values.begin(),
                               bias_cst.float_value_begin(),
                               bias_cst.float_value_end());
      } else {
        return failure();
      }

      int64_t flatten_index = 0;
      for (auto fp_it = filter_cst.float_value_begin(),
                fp_end = filter_cst.float_value_end();
           fp_it != fp_end; ++fp_it) {
        int bias_index = (flatten_index++ / slice_size) % bias_size;

        new_bias_values[bias_index] =
            new_bias_values[bias_index] + *fp_it * cst_value;
      }
      auto new_bias = DenseFPElementsAttr::get(new_bias_type, new_bias_values);
      auto new_bias_op =
          rewriter.create<ConstOp>(fc_op.getLoc(), new_bias_type, new_bias);
      fc_op.setOperand(0, binary_op->getOperand(0));
      fc_op.setOperand(2, new_bias_op);
    } else if (llvm::isa<MulOp>(binary_op) || llvm::isa<DivOp>(binary_op)) {
      // The fusion of mul/div is actually applying the following
      // transformation:
      // w * (x ' c) + b => (w ' c) x + b
      // so we have to update the weight.
      bool is_mul = llvm::isa<MulOp>(binary_op);
      auto new_filter =
          filter_cst.mapValues(filter_type.getElementType(), [&](APFloat it) {
            return (is_mul ? it * cst_value : it / cst_value).bitcastToAPInt();
          });
      // We recreate the constant op in case it is shared by the other ops. This
      // might increase the model size.
      auto new_filter_op = rewriter.create<ConstOp>(
          fc_op.getLoc(), filter.getType(), new_filter);
      fc_op.setOperand(0, binary_op->getOperand(0));
      if (fc_op.filter() != filter) {
        // This filter goes through quantize and dequantize ops. Then we just
        // need to update the weight to the quantize op.
        filter.replaceAllUsesWith(new_filter_op);
      } else {
        // This filter doesn't go through quantize and dequantize ops, Then
        // we update the weight of the affine op directly.
        fc_op.setOperand(1, new_filter_op);
      }
    } else {
      return failure();
    }
    return success();
  }

 private:
  // Returns the dimension length of the channel dimension and also the slide
  // size by each position in the channel dimension accordingly. tfl.conv2d and
  // tfl.fully_connected has heading channel dimension, but tfl.depthwise_conv2d
  // has tailing channel dimension. This function is to provide a utility to
  // create the above information from the op property.
  static std::pair<int64_t, int64_t> GetBiasDimAndSliceSize(
      ArrayRef<int64_t> filter_shape, AffineOpType op) {
    // Channel dimension index is specified as op property
    auto channel_index_iter = filter_shape.begin();
    std::advance(channel_index_iter, op.GetChannelDimIndex());
    // The slide size is the size of the data in higher dimensions.
    int64_t slice_size =
        std::accumulate(std::next(channel_index_iter), filter_shape.end(), 1,
                        std::multiplies<int64_t>());
    return {*channel_index_iter, slice_size};
  }
};

struct ConvertTrivialTransposeOpToReshapeOp
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp transpose_op,
                                PatternRewriter &rewriter) const override {
    auto input_type = transpose_op.input().getType().cast<ShapedType>();
    auto output_type = transpose_op.output().getType().cast<ShapedType>();
    // It's possible to know if the transformation is safe only if the input
    // & output shapes are fully known and permutation is a constant.
    if (!input_type.hasStaticShape() || !output_type.hasStaticShape())
      return failure();
    Value perm = transpose_op.perm();
    DenseElementsAttr perm_values_attr;
    if (!matchPattern(perm, m_Constant(&perm_values_attr))) return failure();

    auto input_shape = input_type.getShape();
    SmallVector<int64_t, 8> perm_values;
    for (const auto &dim : perm_values_attr.getIntValues())
      perm_values.push_back(dim.getSExtValue());

    // This should never happen unless the input graph is malformed.
    if (input_shape.size() != perm_values.size()) {
      transpose_op.emitError(
          "TransposeOP has inconsistent input and perm values.");
    }

    SmallVector<int, 8> old_major_index_ordering;
    SmallVector<int, 8> new_major_index_ordering;
    for (int i = 0; i < input_shape.size(); i++) {
      if (input_shape[i] != 1) {
        old_major_index_ordering.push_back(i);
      }

      if (input_shape[perm_values[i]] != 1) {
        new_major_index_ordering.push_back(perm_values[i]);
      }
    }
    if (old_major_index_ordering != new_major_index_ordering) {
      return failure();
    }

    // Rewrite.
    Location loc = transpose_op.getLoc();

    SmallVector<int32_t, 8> output_shape_values;
    for (auto dim : output_type.getShape()) {
      output_shape_values.push_back(dim);
    }
    auto type = mlir::RankedTensorType::get(output_shape_values.size(),
                                            rewriter.getIntegerType(32));
    auto new_shape_attr =
        mlir::DenseIntElementsAttr::get(type, output_shape_values);
    auto new_shape = rewriter.create<TF::ConstOp>(loc, new_shape_attr);

    rewriter.replaceOpWithNewOp<TFL::ReshapeOp>(
        transpose_op, transpose_op.output().getType(), transpose_op.input(),
        new_shape);

    return success();
  }
};

using FuseBinaryOpToFollowingFullyConnected =
    FuseBinaryOpToFollowingAffineOp<FullyConnectedOp>;
using FuseBinaryOpToFollowingDepthwiseConv2D =
    FuseBinaryOpToFollowingAffineOp<DepthwiseConv2DOp>;
using FuseBinaryOpToFollowingConv2D = FuseBinaryOpToFollowingAffineOp<Conv2DOp>;

void Optimize::runOnFunction() {
  OwningRewritePatternList patterns;
  auto *ctx = &getContext();
  auto func = getFunction();

  // Potentially the binary ops might be fused together, like hard_swish, thus
  // we explore these potentially first and then fuse the binary ops with the
  // following ops in a second pattern match.
  TFL::populateWithGenerated(ctx, &patterns);
  patterns.insert<FuseFullyConnectedAndAdd,
                  FuseFullyConnectedAndReluX<TFL::ReluOp, kRelu>,
                  FuseFullyConnectedAndReluX<TFL::Relu6Op, kRelu6>,
                  FuseFullyConnectedAndReluX<TFL::Relu1Op, kRelu1>,
                  FuseFullyConnectedAndMul>(ctx);
  applyPatternsAndFoldGreedily(func, patterns);

  // Fuse the binary ops with the following ops.
  patterns.insert<
      FuseBinaryOpToFollowingConv2D, FuseBinaryOpToFollowingDepthwiseConv2D,
      FuseBinaryOpToFollowingFullyConnected, FuseConv2DAndMulWithQDQs,
      FuseDepthwiseConv2DAndMulWithQDQs, ConvertTrivialTransposeOpToReshapeOp>(
      ctx);
  applyPatternsAndFoldGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizePass() {
  return std::make_unique<Optimize>();
}

static PassRegistration<Optimize> pass(
    "tfl-optimize", "Optimize within the TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
