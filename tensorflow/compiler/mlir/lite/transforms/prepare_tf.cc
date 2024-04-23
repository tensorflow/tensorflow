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

// This transformation pass prepares for legalization to the TFLite dialect by
// converting operations in TensorFlow dialect into operations that can be
// legalized to TensorFlow Lite dialect with simple replacements.  The newly
// created operations are in the TensorFlow dialect if the operation can be
// represented using a TensorFlow op.  Otherwise, TensorFlow Lite dialect op is
// used.  For example, Conv2D in TFLite which uses OHWI data format for filters
// is not supported in TensorFlow because TensorFlow requires filters in the
// HWIO data format.
//
// Motivation to prepare for the TFLite legalization before the actual
// legalization is to exploit constant folding opportunities in any newly
// created ops by leveraging constant folding support for the TensorFlow ops.
// This way TFLite can be used as a serialization format only and does not
// require access to the TFLite runtime for optimizations as required by the
// TFLite team.

#include <climits>
#include <cstdint>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/dilated_conv.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/constant_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/fake_quant_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/size_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/einsum.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/unroll_batch_matmul.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verification_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#define DEBUG_TYPE "tf-tfl-legalization"

namespace mlir {
namespace TFL {
namespace {
// Returns a TF_CastOp to I32. This function is used for CastOps that are
// intermediate nodes in a TableGen pattern result. In such a case, the
// destination type is not inferred and must be given explicitly.
//
// Preconditions: The given value must have a ShapedType.
static Value CreateTFCastOpI32(OpBuilder *builder, Location loc, Value x,
                               BoolAttr truncate) {
  auto x_type = x.getType().dyn_cast_or_null<ShapedType>();
  if (!x_type) llvm_unreachable("unsupported type");
  Type type = x_type.clone(builder->getI32Type());
  return builder->create<TF::CastOp>(loc, type, x, truncate);
}
}  // namespace

//===----------------------------------------------------------------------===//
// The actual PrepareTF Pass.
//
// TODO(hinsu): Add and use TensorFlow dialect ops for the ops created in this
// pass.
namespace {
#define GEN_PASS_DEF_PREPARETFPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Prepare TF operations in functions for subsequent legalization.
class PrepareTFPass : public impl::PrepareTFPassBase<PrepareTFPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareTFPass)

  PrepareTFPass() = default;
  PrepareTFPass(const PrepareTFPass &) {}
  explicit PrepareTFPass(bool unfold_batch_matmul,
                         bool allow_bf16_and_f16_type_legalization,
                         bool use_fake_quant_num_bits = false) {
    this->unfold_batch_matmul_ = unfold_batch_matmul;
    this->allow_bf16_and_f16_type_legalization_ =
        allow_bf16_and_f16_type_legalization;
    this->use_fake_quant_num_bits_ = use_fake_quant_num_bits;
  }

  void runOnOperation() override;
};

// Transient state for preserving data from match to rewrite
struct ConvertTFConvOpMatchState {
  IntegerAttr dilation_height_factor;
  IntegerAttr dilation_width_factor;
  StringAttr padding;
  IntegerAttr stride_height;
  IntegerAttr stride_width;
};

// Templated class for declaring a converter from some TensorFlow convolution
// op into its counterpart in TensorFlow Lite.
//
// The `ConcreteType` deriving from this template must provide the following
// method for constructing TensorFlow Lite op:
//
//   TFL::[op] createTFLOp(ConvertTFConvOpMatchState *state,
//                         PatternRewriter &rewriter, Location loc,
//                         Type result_type, Value input,
//                         Value filter, Value bias) const;
//
// And also the following method for getting the dimension for bias tensor:
//
//  int64_t getBiasDim(ArrayRef<int64_t> filterShape) const;
template <typename ConcreteType, typename TFConvOpType>
class ConvertTFConvOp : public RewritePattern {
 public:
  ConvertTFConvOp(MLIRContext *context,
                  bool allow_bf16_and_f16_type_legalization)
      : RewritePattern(TFConvOpType::getOperationName(), 1, context),
        intAttrOne(Builder(context).getI32IntegerAttr(1)),
        allow_bf16_and_f16_type_legalization_(
            allow_bf16_and_f16_type_legalization) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Assumes TensorFlow convolution op is already verified to be
    // in valid form.

    // Match a TFConvOpType under the following conditions:
    // * The 'T' attribute must exist and be of value DT_FLOAT.
    // * The 'data_format' attribute must exist and be of value "NHWC".
    // * The 'strides' attribute must exist and is of the form [1, X, Y, 1].
    // * The 'dilations' attribute is optional, but it must be of the form
    //   [1, X, Y, 1] if exists.

    TFConvOpType tf_op = cast<TFConvOpType>(op);
    if (!TFTypeIsFloat32Tensor(tf_op.getInput()) &&
        !(allow_bf16_and_f16_type_legalization_ &&
          TFTypeIsBFloat16OrHalfTensor(tf_op.getInput())))
      return failure();

    if (!TFDataFormatIsNHWC(op)) return failure();

    IntegerAttr height, width;
    if (!TFIntListIs1XY1(op, "strides", &height, &width)) return failure();

    ConvertTFConvOpMatchState state;
    state.stride_height = height;
    state.stride_width = width;

    if (TFIntListIs1XY1(op, "dilations", &height, &width)) {
      state.dilation_height_factor = height;
      state.dilation_width_factor = width;
    } else {
      // If the 'dilations' attribute is missing, we use the default value (1)
      // for both dilation height and width factor.
      state.dilation_height_factor = intAttrOne;
      state.dilation_width_factor = intAttrOne;
    }

    TFPaddingIsSameOrValid(op, &state.padding);

    // Additionally, we require the filter operand to be of 4-D tensor type so
    // that we can extract info from the shape (e.g., for constructing bias
    // tensor, for setting depth_multiplier attribute, etc.).
    auto filter = tf_op.getFilter();
    auto filter_type = filter.getType().template dyn_cast<RankedTensorType>();
    if (!filter_type || filter_type.getRank() != 4 ||
        !filter_type.hasStaticShape())
      return failure();

    Value input = tf_op.getInput();
    RankedTensorType input_type =
        input.getType().template dyn_cast<RankedTensorType>();
    // Only rank size four input will be only available by the tf.Conv2D
    // operator verification.
    if (!input_type || input_type.isDynamicDim(3)) {
      return failure();
    }
    // Check if the given op is based on grouped convolution.
    // Dim size zero will be verified by the tf.Conv2D operator verification.
    if (input_type.getDimSize(3) % filter_type.getDimSize(2) != 0) {
      return failure();
    }

    // TensorFlow convolution op only has two inputs, while the TFLite one has
    // three, with the bias vector marked as optional. However, TOCO has a
    // dedicated pass, EnsureBiasVectors, to create default bias vectors for all
    // those missing. So we model TFLite convolution op as requiring three
    // inputs to achieve the legalization task of EnsureBiasVector. this
    // requires the filter tensor to have static shape.

    // TODO(antiagainst): also handle the case of tf.Add(tf.[op], <bias>)

    // Get a splat zero tensor with the expected dimension for the bias tensor
    auto elem_type = filter_type.getElementType();
    auto bias_dim = static_cast<const ConcreteType *>(this)->getBiasDim(
        filter_type.getShape());
    auto bias_type =
        tensorflow::GetTypeFromTFTensorShape({bias_dim}, elem_type);
    auto bias_attr = rewriter.getZeroAttr(bias_type);
    auto bias =
        rewriter.create<TF::ConstOp>(op->getLoc(), bias_type, bias_attr);

    if (op->getAttrOfType<StringAttr>("padding").getValue() == "EXPLICIT") {
      // Add Const op for padding value.
      ArrayRef<Attribute> padding_attr_array =
          op->getAttrOfType<ArrayAttr>("explicit_paddings").getValue();

      auto get_int = [](Attribute attr) {
        return attr.template cast<IntegerAttr>().getInt();
      };

      SmallVector<int32_t> padding_values(padding_attr_array.size());
      for (int i = 0; i < padding_attr_array.size(); i++) {
        padding_values[i] =
            static_cast<int32_t>(get_int(padding_attr_array[i]));
      }

      RankedTensorType padding_attr_type = tensorflow::GetTypeFromTFTensorShape(
          {filter_type.getRank(), 2}, rewriter.getIntegerType(32));
      auto padding_attr =
          mlir::DenseIntElementsAttr::get(padding_attr_type, padding_values);

      auto padding_const =
          rewriter.create<TF::ConstOp>(op->getLoc(), padding_attr);

      // Add Pad op.
      auto pad_output_type = UnrankedTensorType::get(elem_type);
      input = rewriter.create<TF::PadOp>(op->getLoc(), pad_output_type, input,
                                         padding_const);

      // Set Conv padding to `VALID` since padding has been handled by Pad op.
      state.padding = rewriter.getStringAttr("VALID");
    }
    auto conv_op = static_cast<const ConcreteType *>(this)->createTFLOp(
        &state, rewriter, op->getLoc(), tf_op.getType(), input, filter, bias);

    rewriter.replaceOp(op, conv_op.getResult());
    return success();
  }

  const IntegerAttr intAttrOne;

 private:
  bool allow_bf16_and_f16_type_legalization_;
};

class ConvertTFConv2D : public ConvertTFConvOp<ConvertTFConv2D, TF::Conv2DOp> {
 public:
  using BaseType = ConvertTFConvOp<ConvertTFConv2D, TF::Conv2DOp>;

  ConvertTFConv2D(MLIRContext *context, bool allow_bf16_type_legalization)
      : BaseType(context, allow_bf16_type_legalization) {}

  int64_t getBiasDim(ArrayRef<int64_t> filterShape) const {
    return filterShape.back();
  }

  TFL::Conv2DOp createTFLOp(ConvertTFConvOpMatchState *state,
                            PatternRewriter &rewriter, Location loc,
                            Type result_type, Value input, Value filter,
                            Value bias) const {
    filter = legalizeFilter(rewriter, loc, filter);
    return rewriter.create<TFL::Conv2DOp>(
        loc, result_type, input, filter, bias,
        /*dilation_h_factor=*/state->dilation_height_factor,
        /*dilation_w_factor=*/state->dilation_width_factor,
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
        /*padding=*/state->padding,
        /*stride_h=*/state->stride_height,
        /*stride_w=*/state->stride_width);
  }

 private:
  // Legalize the given filter by converting it from TensorFlow filter data
  // format HWIO to TFLite Conv2D op filter data format OHWI and return Value
  // for the converted filter.  Requires that filter is verified by the match
  // method that it is a 4-D RankedTensorType.
  Value legalizeFilter(PatternRewriter &rewriter, Location loc,
                       Value filter) const {
    // Create a constant op for HWIO to OHWI transpose permutation.
    SmallVector<int, 4> perm = {3, 0, 1, 2};
    auto perm_type = tensorflow::GetTypeFromTFTensorShape(
        {static_cast<int>(perm.size())}, rewriter.getIntegerType(32));
    auto perm_attr =
        DenseElementsAttr::get(perm_type, llvm::ArrayRef<int>(perm));
    auto perm_op = rewriter.create<TF::ConstOp>(loc, perm_type, perm_attr);

    // Create tensor type for the transpose result.
    auto filter_type = filter.getType().cast<RankedTensorType>();
    auto result_shape =
        llvm::to_vector<4>(llvm::map_range(perm, [filter_type](int64_t dim) {
          return filter_type.getDimSize(dim);
        }));
    auto elem_type = filter_type.getElementType();
    auto result_type =
        tensorflow::GetTypeFromTFTensorShape(result_shape, elem_type);

    return rewriter.create<TF::TransposeOp>(loc, result_type, filter, perm_op);
  }
};

class ConvertTFDepthwiseConv2dNative
    : public ConvertTFConvOp<ConvertTFDepthwiseConv2dNative,
                             TF::DepthwiseConv2dNativeOp> {
 public:
  using BaseType = ConvertTFConvOp<ConvertTFDepthwiseConv2dNative,
                                   TF::DepthwiseConv2dNativeOp>;

  ConvertTFDepthwiseConv2dNative(MLIRContext *context,
                                 bool allow_bf16_type_legalization)
      : BaseType(context, allow_bf16_type_legalization) {}

  int64_t getBiasDim(ArrayRef<int64_t> filterShape) const {
    return filterShape[2] * filterShape[3];
  }

  TFL::DepthwiseConv2DOp createTFLOp(ConvertTFConvOpMatchState *state,
                                     PatternRewriter &rewriter, Location loc,
                                     Type result_type, Value input,
                                     Value filter, Value bias) const {
    // Compared to tfl.conv_2d, tfl.depthwise_conv_2d has an additional
    // 'depth_multiplier' attribute. However, tf.DepthwiseConv2dNative does not
    // have a corresponding 'depth_multiplier' attribute; the multiplier is the
    // fourth dimension in the 4-D filter tensor. We query the multiplier from
    // tf.DepthwiseConv2dNative and set it as the attribute value accordingly.
    auto multiplier = filter.getType().cast<RankedTensorType>().getDimSize(3);

    filter = legalizeFilter(rewriter, loc, filter);
    return rewriter.create<TFL::DepthwiseConv2DOp>(
        loc, result_type, input, filter, bias,
        /*dilation_h_factor=*/state->dilation_height_factor,
        /*dilation_w_factor=*/state->dilation_width_factor,
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
        /*padding=*/state->padding,
        /*stride_h=*/state->stride_height,
        /*stride_w=*/state->stride_width,
        /*depth_multiplier=*/rewriter.getI32IntegerAttr(multiplier));
  }

 private:
  /// Legalize the given filter by converting it from TensorFlow filter data
  /// format to TFLite DepthwiseConv2D op filter data format and return Value
  /// for the converted filter.  TensorFlow filter data format is
  /// [filter_height, filter_width, in_channels, channel_multiplier] and TFLite
  /// filter data format is [1, filter_height, filter_width, out_channels].
  /// Requires that filter is verified by the match method that it is a 4-D
  /// RankedTensorType.
  Value legalizeFilter(PatternRewriter &rewriter, Location loc,
                       Value filter) const {
    auto filter_type = filter.getType().cast<RankedTensorType>();
    auto filterShape = filter_type.getShape();
    SmallVector<int64_t, 4> result_shape = {1, filterShape[0], filterShape[1],
                                            filterShape[2] * filterShape[3]};
    auto elem_type = filter_type.getElementType();
    auto result_type =
        tensorflow::GetTypeFromTFTensorShape(result_shape, elem_type);
    // TensorFlow Lite `Reshape` op only support int32 shape tensor currently.
    auto shape_type =
        tensorflow::GetTypeFromTFTensorShape({4}, rewriter.getIntegerType(32));
    SmallVector<Attribute, 4> result_shape_data(4);
    for (int i = 0; i < 4; ++i) {
      auto size = result_shape[i];
      result_shape_data[i] =
          rewriter.getI32IntegerAttr(ConvertToTfliteSize(size));
    }
    auto shape_attr = DenseElementsAttr::get(shape_type, result_shape_data);
    auto shape = rewriter.create<TF::ConstOp>(loc, shape_type, shape_attr);

    return rewriter.create<TF::ReshapeOp>(loc, result_type, filter, shape);
  }
};

// StridedSlice can have complicated attributes like begin_axis_mask,
// end_axis_mask, ellipsis_axis_mask, new_axis_mask, shrink_axis_mask. These
// masks will complicate the strided_slice computation logic, we can simplify
// the logic by inserting a reshape op to pad the inputs so strided_slice can
// be easier to handle.
//
// So the graph may looks like below:
//   original_input -> strided_slice -> output
//      (transforms)
//   original_input -> reshape -> strided_slice -> output
//
// And the new shape is computed based on the masks.
//
// An example for new_axis_mask. say the new_axis_mask is 9 which represents
// [1 0 0 1], and that means we're inserting two new axes at 0 & 3 dim, so
// if original shape is [2, 3], now we reshape that into [1, 2, 3, 1].
struct ConvertTFStridedSlice : public RewritePattern {
  explicit ConvertTFStridedSlice(MLIRContext *context)
      : RewritePattern(TF::StridedSliceOp::getOperationName(), 2, context) {}

  LogicalResult RewriteNewAxisMask(Operation *op,
                                   PatternRewriter &rewriter) const {
    TF::StridedSliceOp strided_slice_op = llvm::cast<TF::StridedSliceOp>(op);
    uint64_t new_axis_mask = strided_slice_op.getNewAxisMask();

    if (strided_slice_op.getEllipsisMask() != 0) {
      // Ellipsis mask should have been lowered-away prior to invoking this
      // function.
      op->emitError() << "encountered a logical error";
      return failure();
    }

    // Insert a new reshape op.
    Value original_input = strided_slice_op.getInput();
    RankedTensorType original_input_type =
        original_input.getType().dyn_cast<RankedTensorType>();
    if (!original_input_type) {
      return failure();
    }

    const ArrayRef<int64_t> &original_input_shape =
        original_input_type.getShape();
    SmallVector<int64_t, 4> revised_shape;
    int index = 0;
    const int original_input_rank = original_input_shape.size();
    while (index < original_input_rank || new_axis_mask) {
      if (new_axis_mask & 1) {
        revised_shape.emplace_back(1);
      } else {
        revised_shape.emplace_back(original_input_shape[index++]);
      }
      new_axis_mask >>= 1;
    }

    if (failed(TF::VerifyShapeOfReshapeOp(revised_shape))) return failure();

    const int dim_size = revised_shape.size();
    Location loc = strided_slice_op.getLoc();
    auto shape_type = tensorflow::GetTypeFromTFTensorShape(
        {dim_size}, rewriter.getIntegerType(32));
    SmallVector<Attribute, 4> result_shape_data(dim_size);
    for (int i = 0; i < dim_size; ++i) {
      auto size = revised_shape[i];
      result_shape_data[i] =
          rewriter.getI32IntegerAttr(ConvertToTfliteSize(size));
    }

    auto shape_attr = DenseElementsAttr::get(shape_type, result_shape_data);
    auto shape =
        rewriter.create<arith::ConstantOp>(loc, shape_type, shape_attr);
    auto revised_output_type = tensorflow::GetTypeFromTFTensorShape(
        revised_shape, original_input_type.getElementType());
    TF::ReshapeOp reshape = rewriter.create<TF::ReshapeOp>(
        loc, revised_output_type, original_input, shape);

    // Replace the original strided_slice.
    uint64_t revised_begin_mask = strided_slice_op.getBeginMask();
    uint64_t revised_end_mask = strided_slice_op.getEndMask();
    // Since we expand the dims, we need to apply them to the begin_mask &
    // end_mask.
    revised_begin_mask |= strided_slice_op.getNewAxisMask();
    revised_end_mask |= strided_slice_op.getNewAxisMask();

    // Enforce operator precedence.
    uint64_t revised_shrink_axis_mask = strided_slice_op.getShrinkAxisMask() &
                                        ~strided_slice_op.getNewAxisMask();

    auto attribute_type = rewriter.getIntegerType(64);
    rewriter.replaceOpWithNewOp<TF::StridedSliceOp>(
        op, strided_slice_op.getType(), reshape, strided_slice_op.getBegin(),
        strided_slice_op.getEnd(), strided_slice_op.getStrides(),
        rewriter.getIntegerAttr(attribute_type, revised_begin_mask),
        rewriter.getIntegerAttr(attribute_type, revised_end_mask),
        rewriter.getIntegerAttr(attribute_type,
                                strided_slice_op.getEllipsisMask()),
        rewriter.getI64IntegerAttr(0),
        rewriter.getIntegerAttr(attribute_type, revised_shrink_axis_mask));
    return success();
  }

  LogicalResult RewriteEllipsisMask(Operation *op,
                                    PatternRewriter &rewriter) const {
    TF::StridedSliceOp strided_slice_op = llvm::cast<TF::StridedSliceOp>(op);

    uint64_t ellipsis_mask = strided_slice_op.getEllipsisMask();
    uint64_t shrink_axis_mask = strided_slice_op.getShrinkAxisMask();
    uint64_t new_axis_mask = strided_slice_op.getNewAxisMask();

    // Enforce operator precedence.
    shrink_axis_mask &= ~ellipsis_mask;
    new_axis_mask &= ~ellipsis_mask;

    DenseIntElementsAttr begin_dense_elem_attr;
    Value begin = strided_slice_op.getBegin();
    auto begin_ranked_attr_type = begin.getType().dyn_cast<RankedTensorType>();
    if (!begin_ranked_attr_type ||
        !matchPattern(begin, m_Constant(&begin_dense_elem_attr))) {
      return failure();
    }

    DenseIntElementsAttr end_dense_elem_attr;
    Value end = strided_slice_op.getEnd();
    auto end_ranked_attr_type = end.getType().dyn_cast<RankedTensorType>();
    if (!end_ranked_attr_type ||
        !matchPattern(end, m_Constant(&end_dense_elem_attr))) {
      return failure();
    }

    DenseIntElementsAttr stride_dense_elem_attr;
    Value stride = strided_slice_op.getStrides();
    auto stride_ranked_attr_type =
        stride.getType().dyn_cast<RankedTensorType>();
    if (!stride_ranked_attr_type ||
        !matchPattern(stride, m_Constant(&stride_dense_elem_attr))) {
      return failure();
    }

    Value input = strided_slice_op.getInput();
    RankedTensorType input_type = input.getType().dyn_cast<RankedTensorType>();
    if (!input_type) {
      return failure();
    }
    const ArrayRef<int64_t> input_shape = input_type.getShape();

    const int input_size = input_shape.size();

    RankedTensorType begin_type = begin.getType().cast<RankedTensorType>();
    const ArrayRef<int64_t> begin_shape = begin_type.getShape();
    const int begin_dim = begin_shape.size();

    if (begin_dim != 1) return failure();

    // The ellipsis fill might exceed the current output shape because we are
    // also taking account of any to-be-inserted new axes.
    const int ellipsis_filled_dim_size =
        input_size - begin_shape[0] + 1 + absl::popcount(new_axis_mask);

    int64_t begin_mask = strided_slice_op.getBeginMask();
    int64_t end_mask = strided_slice_op.getEndMask();
    int64_t revised_begin_mask = 0;
    int64_t revised_end_mask = 0;
    int64_t revised_shrink_axis_mask = 0;
    int64_t revised_new_axis_mask = 0;

    SmallVector<int32_t, 4> padded_begin;
    SmallVector<int32_t, 4> padded_end;
    SmallVector<int32_t, 4> padded_stride;

    // Before the ellipsis.
    int index = 0;
    int new_index = 0;
    while (((ellipsis_mask >> index) & 1) == 0) {
      padded_begin.push_back(begin_dense_elem_attr.getValues<int32_t>()[index]);
      padded_end.push_back(end_dense_elem_attr.getValues<int32_t>()[index]);
      padded_stride.push_back(
          stride_dense_elem_attr.getValues<int32_t>()[index]);
      if ((begin_mask >> index) & 1) revised_begin_mask |= (1 << new_index);
      if ((end_mask >> index) & 1) revised_end_mask |= (1 << new_index);
      if ((shrink_axis_mask >> index) & 1)
        revised_shrink_axis_mask |= (1 << new_index);

      if ((new_axis_mask >> index) & 1)
        revised_new_axis_mask |= (1 << new_index);

      ++index;
      ++new_index;
    }

    // Ellipsis.
    for (; new_index < index + ellipsis_filled_dim_size; ++new_index) {
      revised_begin_mask |= (1 << new_index);
      revised_end_mask |= (1 << new_index);

      // Mimic the begin/end/strides mask behavior.
      padded_begin.push_back(0);
      padded_end.push_back(0);
      padded_stride.push_back(1);
    }

    // Account for ellipsis mask.
    ++index;

    // After the ellipsis.
    for (; index < begin_shape[0];) {
      padded_begin.push_back(begin_dense_elem_attr.getValues<int32_t>()[index]);
      padded_end.push_back(end_dense_elem_attr.getValues<int32_t>()[index]);
      padded_stride.push_back(
          stride_dense_elem_attr.getValues<int32_t>()[index]);

      if ((begin_mask >> index) & 1) revised_begin_mask |= (1 << new_index);
      if ((end_mask >> index) & 1) revised_end_mask |= (1 << new_index);
      if ((shrink_axis_mask >> index) & 1)
        revised_shrink_axis_mask |= (1 << new_index);
      if ((new_axis_mask >> index) & 1)
        revised_new_axis_mask |= (1 << new_index);

      ++index;
      ++new_index;
    }

    auto attribute_type = rewriter.getIntegerType(64);

    int full_dim_count = padded_begin.size();
    auto type = tensorflow::GetTypeFromTFTensorShape(
        {full_dim_count}, rewriter.getIntegerType(32));

    auto begin_attr = DenseElementsAttr::get<int32_t>(type, padded_begin);
    auto begin_op =
        rewriter.create<arith::ConstantOp>(op->getLoc(), type, begin_attr);
    auto end_attr = DenseElementsAttr::get<int32_t>(type, padded_end);
    auto end_op =
        rewriter.create<arith::ConstantOp>(op->getLoc(), type, end_attr);
    auto stride_attr = DenseElementsAttr::get<int32_t>(type, padded_stride);
    auto stride_op =
        rewriter.create<arith::ConstantOp>(op->getLoc(), type, stride_attr);

    rewriter.replaceOpWithNewOp<TF::StridedSliceOp>(
        op, strided_slice_op.getType(), input, begin_op.getResult(),
        end_op.getResult(), stride_op.getResult(),
        rewriter.getIntegerAttr(attribute_type, revised_begin_mask),
        rewriter.getIntegerAttr(attribute_type, revised_end_mask),
        /*ellipsis_mask=*/rewriter.getI64IntegerAttr(0),
        rewriter.getIntegerAttr(attribute_type, revised_new_axis_mask),
        rewriter.getIntegerAttr(attribute_type, revised_shrink_axis_mask));

    return success();
  }

  void PadStridedSliceAttributeArray(DenseIntElementsAttr dense_elem_attr,
                                     SmallVectorImpl<int32_t> &val,
                                     SmallVectorImpl<int32_t> &padded_val,
                                     ArrayRef<int32_t> padding_val,
                                     int *mask) const {
    for (const auto &idx : dense_elem_attr.getValues<APInt>()) {
      val.push_back(idx.getSExtValue());
      padded_val.push_back(idx.getSExtValue());
    }
    int attr_dim_count = val.size();
    int full_dim_count = padding_val.size();
    for (int i = attr_dim_count; i < full_dim_count; ++i) {
      padded_val.push_back(padding_val[i]);
      if (mask) *mask |= 1 << i;
    }
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    TF::StridedSliceOp strided_slice_op = llvm::cast<TF::StridedSliceOp>(op);

    // Handle ellipsis mask.
    if (strided_slice_op.getEllipsisMask() != 0) {
      return RewriteEllipsisMask(strided_slice_op, rewriter);
    }

    // Handle new axis mask.
    if (strided_slice_op.getNewAxisMask() != 0) {
      return RewriteNewAxisMask(strided_slice_op, rewriter);
    }

    auto ranked_input_type =
        strided_slice_op.getInput().getType().dyn_cast<RankedTensorType>();
    if (!ranked_input_type) {
      return failure();
    }

    auto begin_attr = strided_slice_op.getBegin();
    auto end_attr = strided_slice_op.getEnd();
    auto strides_attr = strided_slice_op.getStrides();

    auto begin_attr_type = begin_attr.getType().dyn_cast<RankedTensorType>();
    auto end_attr_type = end_attr.getType().dyn_cast<RankedTensorType>();
    auto strides_attr_type =
        strides_attr.getType().dyn_cast<RankedTensorType>();

    DenseIntElementsAttr begin_elem_attr;
    DenseIntElementsAttr end_elem_attr;
    DenseIntElementsAttr strides_elem_attr;

    if (!begin_attr_type ||
        !matchPattern(begin_attr, m_Constant(&begin_elem_attr))) {
      return failure();
    }
    if (!end_attr_type || !matchPattern(end_attr, m_Constant(&end_elem_attr))) {
      return failure();
    }
    if (!strides_attr_type ||
        !matchPattern(strides_attr, m_Constant(&strides_elem_attr))) {
      return failure();
    }

    SmallVector<int32_t, 4> begin, end, strides;
    SmallVector<int32_t, 4> padded_begin, padded_end, padded_strides;

    int num_input_dims = ranked_input_type.getRank();
    SmallVector<int32_t, 4> padding_begin(num_input_dims, 0);
    auto input_shape = ranked_input_type.getShape();
    SmallVector<int32_t, 4> padding_end(input_shape.begin(), input_shape.end());
    SmallVector<int32_t, 4> padding_strides(num_input_dims, 1);

    int begin_mask = strided_slice_op.getBeginMask();
    int end_mask = strided_slice_op.getEndMask();

    PadStridedSliceAttributeArray(begin_elem_attr, begin, padded_begin,
                                  padding_begin, &begin_mask);
    PadStridedSliceAttributeArray(end_elem_attr, end, padded_end, padding_end,
                                  &end_mask);
    PadStridedSliceAttributeArray(strides_elem_attr, strides, padded_strides,
                                  padding_strides, nullptr);

    if (begin == padded_begin && end == padded_end &&
        strides == padded_strides &&
        begin_mask == strided_slice_op.getBeginMask() &&
        end_mask == strided_slice_op.getEndMask()) {
      return failure();
    }

    auto begin_end_type = tensorflow::GetTypeFromTFTensorShape(
        {num_input_dims}, rewriter.getIntegerType(32));
    auto new_begin_attr = rewriter.create<arith::ConstantOp>(
        op->getLoc(), begin_end_type,
        DenseElementsAttr::get<int32_t>(begin_end_type, padded_begin));
    auto new_end_attr = rewriter.create<arith::ConstantOp>(
        op->getLoc(), begin_end_type,
        DenseElementsAttr::get<int32_t>(begin_end_type, padded_end));
    auto strides_type = tensorflow::GetTypeFromTFTensorShape(
        {static_cast<int64_t>(padded_strides.size())},
        rewriter.getIntegerType(32));
    auto new_strides_attr = rewriter.create<arith::ConstantOp>(
        op->getLoc(), strides_type,
        DenseElementsAttr::get<int32_t>(strides_type, padded_strides));

    auto attribute_type = rewriter.getIntegerType(64);
    rewriter.replaceOpWithNewOp<TF::StridedSliceOp>(
        op, strided_slice_op.getOutput().getType(), strided_slice_op.getInput(),
        new_begin_attr, new_end_attr, new_strides_attr,
        rewriter.getIntegerAttr(attribute_type, begin_mask),
        rewriter.getIntegerAttr(attribute_type, end_mask),
        rewriter.getIntegerAttr(attribute_type,
                                strided_slice_op.getEllipsisMask()),
        rewriter.getIntegerAttr(attribute_type,
                                strided_slice_op.getNewAxisMask()),
        rewriter.getIntegerAttr(attribute_type,
                                strided_slice_op.getShrinkAxisMask()));

    return success();
  }
};


// The below pattern is equivalent to the DRR rule below
// The checks are dependent on generated values, so we can't add
// the checks on intermediate values, ideally we should find equivalent
// checks that guarantees the resultant ops are valid.
// The extra conditions are the broadcasting conditions.
//
// The pattern lower FusedBatchNormV3 to arithmetic ops.
// Specifically, performs the following calculation:
//
//   (x - mean) * scale / sqrt(variance + epsilon) + offset
//
// Let multiplier = scale / sqrt(variance + epsilon),
// to compute
//   (x - mean) * scale / sqrt(variance + epsilon) + offset,
// is then to compute
//   (x * multiplier) + (offset - mean * multiplier).
//
// def : Pattern<
//     (TF_FusedBatchNormV3Op:$root
//         $x, $scale, $offset, $mean, $variance,
//         F32Attr:$epsilon, $exponential_avg_factor,
//         $data_format, FalseBoolAttr:$is_training),
//     [(TF_AddOp
//         (TF_MulOp
//             $x,
//             (TF_MulOp:$multiplier
//                 $scale,
//                 (TF_RsqrtOp
//                     (TF_AddOp $variance,
//                               (TF_ConstOp $epsilon))))),
//         (TF_SubOp $offset, (TF_MulOp $mean, $multiplier))),
//    // We already guaranteed that the last five results have no use so it does
//    // not matter what value we provide here for replacement.
//      /*batch_mean=*/(replaceWithValue $x),
//      /*batch_variance=*/(replaceWithValue $x),
//      /*reserve_space_1=*/(replaceWithValue $x),
//      /*reserve_space_2=*/(replaceWithValue $x),
//      /*reserve_space_3=*/(replaceWithValue $x)],
//     [(HasNoUseOf:$root__1), (HasNoUseOf:$root__2),
//      (HasNoUseOf:$root__3), (HasNoUseOf:$root__4),
//      (HasNoUseOf:$root__5), (AreBroadcastableTypes $multiplier, $x)]>;
//
// When is_training is set to true, the given variance and mean are not used.
// In above calculation, they are replaced by new values. These new mean and
// variance are calculated as following:
// new_mean = mean(x, axis=[0, 1, 2])
// new_variance = mean(squared_difference(x, new_mean), axis=[0, 1, 2])
//
// The DDR rule for the is_training equals true case is as following:
// def : Pattern<
//     (TF_FusedBatchNormV3Op:$root
//         $x, $scale, $offset, $mean, $variance,
//         F32Attr:$epsilon, $exponential_avg_factor,
//         $data_format, FalseBoolAttr:$is_training),
//     [(TF_AddOp
//         (TF_MulOp
//             $x,
//             (TF_MulOp:$multiplier
//                 $scale,
//                 (TF_RsqrtOp
//                     (TF_AddOp
//                         (TF_MeanOp
//                             (TF_SquaredDifferenceOp $x, $new_mean),
//                             (TF_ConstOp [0,1,2])),
//                         (TF_ConstOp $epsilon))))),
//         (TF_SubOp
//             $offset,
//             (TF_MulOp
//                 (TF_MeanOp $x, (TF_ConstOp [0,1,2])),
//                 $multiplier))),
//    // We already guaranteed that the last five results have no use so it does
//    // not matter what value we provide here for replacement.
//      /*batch_mean=*/(replaceWithValue $x),
//      /*batch_variance=*/(replaceWithValue $x),
//      /*reserve_space_1=*/(replaceWithValue $x),
//      /*reserve_space_2=*/(replaceWithValue $x),
//      /*reserve_space_3=*/(replaceWithValue $x)],
//     [(HasNoUseOf:$root__1), (HasNoUseOf:$root__2),
//      (HasNoUseOf:$root__3), (HasNoUseOf:$root__4),
//      (HasNoUseOf:$root__5), (AreBroadcastableTypes $multiplier, $x)]>;

struct FusedBatchNormV3Pat : public ::mlir::RewritePattern {
  explicit FusedBatchNormV3Pat(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern(
            "tf.FusedBatchNormV3", 1, context,
            {"tf.Add", "tf.Const", "tf.Mul", "tf.Rsqrt", "tf.Sub"}) {}

  ::mlir::LogicalResult matchAndRewrite(
      ::mlir::Operation *fused_batch_norm,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    Operation::operand_range mean(fused_batch_norm->getOperands());
    ::mlir::FloatAttr exponential_avg_factor;
    ::mlir::TF::FusedBatchNormV3Op root;
    Operation::operand_range offset(fused_batch_norm->getOperands());
    Operation::operand_range x(fused_batch_norm->getOperands());
    Operation::operand_range scale(fused_batch_norm->getOperands());
    Operation::operand_range variance(fused_batch_norm->getOperands());
    ::mlir::FloatAttr epsilon;
    ::mlir::BoolAttr is_training;

    // Match
    auto fused_batch_norm_op =
        dyn_cast_or_null<::mlir::TF::FusedBatchNormV3Op>(fused_batch_norm);
    root = fused_batch_norm_op;
    x = fused_batch_norm_op.getODSOperands(0);
    scale = fused_batch_norm_op.getODSOperands(1);
    offset = fused_batch_norm_op.getODSOperands(2);
    mean = fused_batch_norm_op.getODSOperands(3);
    variance = fused_batch_norm_op.getODSOperands(4);

    ::mlir::Value mean_value = (*mean.begin());
    ::mlir::Value variance_value = (*variance.begin());

    if (!TFTypeIsFloat32Tensor(fused_batch_norm_op.getX())) return failure();

    {
      epsilon =
          fused_batch_norm_op->getAttrOfType<::mlir::FloatAttr>("epsilon");
      if (!epsilon)
        epsilon = rewriter.getFloatAttr(rewriter.getF32Type(), 0.0001f);

      if (!(((epsilon.isa<::mlir::FloatAttr>())) &&
            ((epsilon.cast<::mlir::FloatAttr>().getType().isF32())))) {
        return rewriter.notifyMatchFailure(
            fused_batch_norm_op, [&](::mlir::Diagnostic &diag) {
              diag << "op 'tf.FusedBatchNormV3' attribute 'epsilon' failed to "
                      "satisfy constraint: 32-bit float attribute";
            });
      }
    }
    {
      exponential_avg_factor =
          fused_batch_norm_op->getAttrOfType<::mlir::FloatAttr>(
              "exponential_avg_factor");
      if (!exponential_avg_factor)
        exponential_avg_factor =
            rewriter.getFloatAttr(rewriter.getF32Type(), 1.0f);
    }
    if (!TFDataFormatIsNHWC(fused_batch_norm_op) &&
        !TFDataFormatIsNDHWC(fused_batch_norm_op))
      return failure();

    if (!(((*root.getODSResults(1).begin()).use_empty()))) {
      return rewriter.notifyMatchFailure(
          fused_batch_norm_op, [&](::mlir::Diagnostic &diag) {
            diag << "entities '' failed to satisfy constraint: has no use";
          });
    }

    if (!(((*root.getODSResults(2).begin()).use_empty()))) {
      return rewriter.notifyMatchFailure(
          fused_batch_norm_op, [&](::mlir::Diagnostic &diag) {
            diag << "entities '' failed to satisfy constraint: has no use";
          });
    }

    if (!(((*root.getODSResults(3).begin()).use_empty()))) {
      return rewriter.notifyMatchFailure(
          fused_batch_norm_op, [&](::mlir::Diagnostic &diag) {
            diag << "entities '' failed to satisfy constraint: has no use";
          });
    }

    if (!(((*root.getODSResults(4).begin()).use_empty()))) {
      return rewriter.notifyMatchFailure(
          fused_batch_norm_op, [&](::mlir::Diagnostic &diag) {
            diag << "entities '' failed to satisfy constraint: has no use";
          });
    }

    if (!(((*root.getODSResults(5).begin()).use_empty()))) {
      return rewriter.notifyMatchFailure(
          fused_batch_norm_op, [&](::mlir::Diagnostic &diag) {
            diag << "entities '' failed to satisfy constraint: has no use";
          });
    }

    is_training =
        fused_batch_norm_op->getAttrOfType<::mlir::BoolAttr>("is_training");
    auto odsLoc = rewriter.getFusedLoc({fused_batch_norm->getLoc()});

    // We need to make sure input and output shapes are compatible.
    int64_t last_dim = ShapedType::kDynamic;
    {
      auto is_last_dim_compatible = [](const Value &v, int64_t &last_dim) {
        auto v_type = v.getType().dyn_cast_or_null<RankedTensorType>();
        if (!v_type) return true;
        int64_t v_last_dim = v_type.getDimSize(v_type.getRank() - 1);
        if (v_last_dim == ShapedType::kDynamic) return true;
        if (last_dim != ShapedType::kDynamic && v_last_dim != last_dim)
          return false;
        last_dim = v_last_dim;
        return true;
      };

      if (!is_last_dim_compatible(*x.begin(), last_dim) ||
          !is_last_dim_compatible(*scale.begin(), last_dim) ||
          !is_last_dim_compatible(*offset.begin(), last_dim)) {
        return rewriter.notifyMatchFailure(
            fused_batch_norm_op, [&](::mlir::Diagnostic &diag) {
              diag << "Shapes of scale and offset should be 1D and "
                      "compatible with x";
            });
      }

      if (!is_training.getValue()) {
        if (!is_last_dim_compatible(mean_value, last_dim) ||
            !is_last_dim_compatible(variance_value, last_dim)) {
          return rewriter.notifyMatchFailure(
              fused_batch_norm_op, [&](::mlir::Diagnostic &diag) {
                diag << "Shapes of mean and variance should be 1D and "
                        "compatible with x";
              });
        }
      }

      // Check if output shape and input shape are compatible.
      auto x_type = (*x.begin()).getType();
      auto y_type = (*root.getODSResults(0).begin()).getType();
      if (!OpTrait::util::getBroadcastedType(x_type, y_type)) {
        return rewriter.notifyMatchFailure(
            fused_batch_norm_op, [&](::mlir::Diagnostic &diag) {
              diag << "Shapes of x and the first output should be compatible";
            });
      }
    }

    // For training, mean and variance is calculated from input values.
    if (is_training.getValue()) {
      auto input_type = fused_batch_norm_op.getX()
                            .getType()
                            .dyn_cast_or_null<RankedTensorType>();
      if (!input_type || input_type.getRank() != 4) {
        return rewriter.notifyMatchFailure(
            fused_batch_norm_op, [&](::mlir::Diagnostic &diag) {
              diag << "op 'tf.FusedBatchNormV3' that has 'is_training' equals "
                      "True is only supported with input of rank 4";
            });
      }

      ::mlir::TF::ConstOp reduce_dim_op;
      {
        auto reduce_dim_type = tensorflow::GetTypeFromTFTensorShape(
            {3}, rewriter.getIntegerType(32));
        ::mlir::SmallVector<int32_t, 3> reduce_dim_values = {0, 1, 2};
        reduce_dim_op = rewriter.create<TF::ConstOp>(
            odsLoc, ::mlir::DenseIntElementsAttr::get(reduce_dim_type,
                                                      reduce_dim_values));
      }

      auto new_mean_type = tensorflow::GetTypeFromTFTensorShape(
          {last_dim}, rewriter.getF32Type());
      ::mlir::TF::MeanOp mean_op_1;
      {
        ::mlir::Value x_value = (*x.begin());
        mean_op_1 = rewriter.create<TF::MeanOp>(
            odsLoc, new_mean_type, x_value, reduce_dim_op,
            /*keep_dims=*/rewriter.getBoolAttr(false));
      }

      ::mlir::TF::SquaredDifferenceOp square_diff_op;
      {
        ::mlir::Value tblgen_value_0 = (*x.begin());
        ::mlir::Value tblgen_value_1 = (*mean_op_1.getODSResults(0).begin());
        // If x has shape of [b, h, w, c], the result of mean_op_1 will have
        // shape of [c]. Therefore, their shapes are always compatible.
        square_diff_op = rewriter.create<::mlir::TF::SquaredDifferenceOp>(
            odsLoc, tblgen_value_0, tblgen_value_1);
      }

      ::mlir::TF::MeanOp mean_op_2;
      {
        ::mlir::Value input_value = (*square_diff_op.getODSResults(0).begin());
        mean_op_2 = rewriter.create<TF::MeanOp>(
            odsLoc, new_mean_type, input_value, reduce_dim_op,
            /*keep_dims=*/rewriter.getBoolAttr(false));
      }

      mean_value = (*mean_op_1.getODSResults(0).begin());
      variance_value = (*mean_op_2.getODSResults(0).begin());
    }  // End is_training equals true if.

    ::llvm::SmallVector<::mlir::Value, 4> replace_values;
    ::mlir::TF::ConstOp epsilon_const_op;
    {
      epsilon_const_op =
          rewriter.create<::mlir::TF::ConstOp>(odsLoc,
                                               /*value=*/epsilon);
    }
    ::mlir::TF::AddOp add_op_1;
    {
      ::mlir::Value epsilon_value =
          (*epsilon_const_op.getODSResults(0).begin());
      // Multiplying with a constant, no need to check broadcastibility.
      add_op_1 = rewriter.create<::mlir::TF::AddOp>(odsLoc,
                                                    /*x=*/variance_value,
                                                    /*y=*/epsilon_value);
    }
    ::mlir::TF::RsqrtOp rsqrt_op;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      tblgen_values.push_back((*add_op_1.getODSResults(0).begin()));
      rsqrt_op = rewriter.create<::mlir::TF::RsqrtOp>(odsLoc, tblgen_values,
                                                      tblgen_attrs);
    }
    ::mlir::TF::MulOp multiplier;
    {
      ::mlir::Value tblgen_value_0 = (*scale.begin());
      ::mlir::Value tblgen_value_1 = (*rsqrt_op.getODSResults(0).begin());
      multiplier = rewriter.create<::mlir::TF::MulOp>(odsLoc,
                                                      /*x=*/tblgen_value_0,
                                                      /*y=*/tblgen_value_1);
    }
    ::mlir::TF::MulOp mul_op_1;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*multiplier.getODSResults(0).begin());
      mul_op_1 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
                                                    /*x=*/tblgen_value_0,
                                                    /*y=*/tblgen_value_1);
    }
    ::mlir::TF::MulOp mul_op_2;
    {
      ::mlir::Value multiplier_value = (*multiplier.getODSResults(0).begin());
      mul_op_2 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
                                                    /*x=*/mean_value,
                                                    /*y=*/multiplier_value);
    }
    ::mlir::TF::SubOp sub_op;
    {
      ::mlir::Value tblgen_value_0 = (*offset.begin());
      ::mlir::Value tblgen_value_1 = (*mul_op_2.getODSResults(0).begin());
      sub_op = rewriter.create<::mlir::TF::SubOp>(odsLoc,
                                                  /*x=*/tblgen_value_0,
                                                  /*y=*/tblgen_value_1);
    }
    ::mlir::TF::AddOp add_op_2;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs;
      tblgen_values.push_back((*mul_op_1.getODSResults(0).begin()));
      tblgen_values.push_back((*sub_op.getODSResults(0).begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types;
      for (auto v : fused_batch_norm_op.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      add_op_2 = rewriter.create<::mlir::TF::AddOp>(
          odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }
    for (auto v :
         ::llvm::SmallVector<::mlir::Value, 4>{add_op_2.getODSResults(0)}) {
      replace_values.push_back(v);
    }
    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{x}) {
      replace_values.push_back(v);
    }
    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{x}) {
      replace_values.push_back(v);
    }
    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{x}) {
      replace_values.push_back(v);
    }
    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{x}) {
      replace_values.push_back(v);
    }
    for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{x}) {
      replace_values.push_back(v);
    }
    rewriter.replaceOp(fused_batch_norm, replace_values);
    return success();
  };
};

static LogicalResult static_dag_matcher(
    PatternRewriter &rewriter, Operation *op,
    ::llvm::SmallVector<Operation *, 4> &target_ops,
    Operation::operand_range &max, Operation::operand_range &min,
    Operation::operand_range &input, IntegerAttr &num_bits,
    BoolAttr &narrow_range) {
  auto fakequant_op = ::llvm::dyn_cast<TF::FakeQuantWithMinMaxVarsOp>(op);
  if (!(fakequant_op)) {
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "fakequant_op is not TF::FakeQuantWithMinMaxVarsOp type";
    });
  }
  input = fakequant_op.getODSOperands(0);
  min = fakequant_op.getODSOperands(1);
  max = fakequant_op.getODSOperands(2);
  {
    auto target_attr = op->getAttrOfType<IntegerAttr>("num_bits");
    if (!target_attr)
      target_attr = rewriter.getIntegerAttr(rewriter.getIntegerType(64), 8);
    num_bits = target_attr;
  }
  {
    auto target_attr = op->getAttrOfType<BoolAttr>("narrow_range");
    if (!target_attr) target_attr = rewriter.getBoolAttr(false);
    narrow_range = target_attr;
  }
  {
    for (auto user : fakequant_op->getResult(0).getUsers()) {
      if (!absl::c_linear_search(target_ops, user)) {
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
          diag << "Skipping reordering between FakeQuant and "
               << (*target_ops.begin())->getName()
               << ", since there are other ops using the FakeQuant result.";
        });
      }
    }
  }
  return ::mlir::success();
}

// Reorder the FakeQuant operation for specific ops (ReorderOp).
// The transformation pattern looks like below:
//
//    <allowed>               <not allowed>
//
//    fakequant                 fakequant
//        |                         |
//    reorder_op                   / \
//                       reorder_op   other_op
//
//       ||
//       \/
//
//    reorder_op
//        |
//    fakequant
template <typename ReorderOp>
struct ReorderFakeQuantPattern : public RewritePattern {
  explicit ReorderFakeQuantPattern(MLIRContext *context)
      : RewritePattern(
            ReorderOp::getOperationName(), 2, context,
            {"tf.FakeQuantWithMinMaxVars", ReorderOp::getOperationName()}) {}
  LogicalResult findTargetOps(
      ReorderOp &casted_op, PatternRewriter &rewriter,
      Operation::operand_range &max, Operation::operand_range &min,
      Operation::operand_range &input, IntegerAttr &num_bits,
      BoolAttr &narrow_range,
      ::llvm::SmallVector<Operation *, 4> &target_ops) const {
    auto *defining_op = (*casted_op.getODSOperands(0).begin()).getDefiningOp();
    if (!(defining_op)) {
      return rewriter.notifyMatchFailure(casted_op, [&](Diagnostic &diag) {
        diag << "There's no operation that defines operand 0 of casted_op";
      });
    }
    if (failed(static_dag_matcher(rewriter, defining_op, target_ops, max, min,
                                  input, num_bits, narrow_range))) {
      return failure();
    }
    target_ops.push_back(defining_op);
    return success();
  }

  LogicalResult CreateReorderOp(PatternRewriter &rewriter,
                                Operation::operand_range &input,
                                Operation::operand_range &shape,
                                Location &ods_loc,
                                ReorderOp &new_reorder_op) const {
    Value tensor_value = (*input.begin());
    Value shape_value = (*shape.begin());
    new_reorder_op = rewriter.create<ReorderOp>(ods_loc,
                                                /*tensor=*/tensor_value,
                                                /*shape=*/shape_value);
    return success();
  }

  LogicalResult createFakeQuantOp(
      ReorderOp &casted_op, PatternRewriter &rewriter,
      ReorderOp &new_reorder_op, Operation::operand_range &max,
      Operation::operand_range &min, IntegerAttr &num_bits,
      BoolAttr &narrow_range, Location &ods_loc,
      TF::FakeQuantWithMinMaxVarsOp &fakequant_op) const {
    ::llvm::SmallVector<Value, 4> target_values;
    ::llvm::SmallVector<NamedAttribute, 4> target_attrs;
    target_values.push_back((*new_reorder_op.getODSResults(0).begin()));
    target_values.push_back((*min.begin()));
    target_values.push_back((*max.begin()));
    if (auto tmpAttr = num_bits) {
      target_attrs.emplace_back(rewriter.getStringAttr("num_bits"), tmpAttr);
    }
    if (auto tmpAttr = narrow_range) {
      target_attrs.emplace_back(rewriter.getStringAttr("narrow_range"),
                                tmpAttr);
    }
    ::llvm::SmallVector<Type, 4> target_types;
    for (auto v : casted_op.getODSResults(0)) {
      target_types.push_back(v.getType());
    }
    fakequant_op = rewriter.create<TF::FakeQuantWithMinMaxVarsOp>(
        ods_loc, target_types, target_values, target_attrs);
    return success();
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    Operation::operand_range max(op->getOperands());
    Operation::operand_range min(op->getOperands());
    Operation::operand_range input(op->getOperands());
    Operation::operand_range shape(op->getOperands());
    IntegerAttr num_bits;
    BoolAttr narrow_range;
    ::llvm::SmallVector<Operation *, 4> target_ops;

    target_ops.push_back(op);
    ReorderOp old_reorder_op = ::llvm::dyn_cast<ReorderOp>(op);
    if (failed(findTargetOps(old_reorder_op, rewriter, max, min, input,
                             num_bits, narrow_range, target_ops))) {
      return failure();
    }
    shape = old_reorder_op.getODSOperands(1);

    // Rewrite
    auto ods_loc = rewriter.getFusedLoc(
        {target_ops[0]->getLoc(), target_ops[1]->getLoc()});
    ReorderOp new_reorder_op;
    if (failed(
            CreateReorderOp(rewriter, input, shape, ods_loc, new_reorder_op))) {
      return failure();
    }
    ::llvm::SmallVector<Value, 4> target_repl_values;
    TF::FakeQuantWithMinMaxVarsOp new_fakequant_op;
    if (failed(createFakeQuantOp(old_reorder_op, rewriter, new_reorder_op, max,
                                 min, num_bits, narrow_range, ods_loc,
                                 new_fakequant_op))) {
      return failure();
    }
    for (auto v :
         ::llvm::SmallVector<Value, 4>{new_fakequant_op.getODSResults(0)}) {
      target_repl_values.push_back(v);
    }

    rewriter.replaceOp(old_reorder_op, target_repl_values);
    return success();
  };
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_prepare_tf.inc"

// Returns success if all the operations in the `op`'s regions including `op`
// itself are legal in a TFLite pipeline.
LogicalResult ValidateOp(Operation *op) {
  bool has_illegal_ops = false;
  op->walk([&](Operation *op) {
    if (isa<TF::VariableV2Op>(op)) {
      has_illegal_ops = true;
      op->emitOpError() << "is illegal in a TFLite pipeline";
    }
  });

  return failure(has_illegal_ops);
}

// Converts a set of TF2XLA ops into pure TF ops for future legalizations as
// TF2XLA ops aren't supported by later stages.
LogicalResult ConvertTf2XlaOps(func::FuncOp func, MLIRContext *context) {
  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<TF::TensorFlowDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<func::FuncOp>();
  target.addIllegalOp<TF::XlaConvV2Op>();
  target.addIllegalOp<TF::XlaGatherOp>();

  RewritePatternSet patterns(context);
  mhlo::Tf2XlaTypeConverter converter;
  mhlo::PopulateLegalizeTfWithTf2XlaPatterns("XLA_CPU_JIT", patterns, context,
                                             converter);
  mhlo::PopulateLegalizeTfPatterns(context, &patterns);
  mlir::odml::PopulateLegalizeHloToTfPatterns(&patterns, context);
  mhlo::GatherOp::getCanonicalizationPatterns(patterns, context);

  return applyPartialConversion(func, target, std::move(patterns));
}

// Convert rfft to rfft2d.
// The transformation pattern looks like below:
//
//    input     fft_len
//     \      /
//     rfft
//
//     ||
//     \/
//
//   input       fft_len
//    \            /
//   expand_dim    concat with [1] at the front
//      \         /
//     rfft_2d
//       |
//     squeeze
struct ConvertRfftToRfft2d : public RewritePattern {
  explicit ConvertRfftToRfft2d(MLIRContext *context)
      : RewritePattern(TF::RFFTOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto rfft_op = dyn_cast<TF::RFFTOp>(op);

    auto input = rfft_op.getInput();
    auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
    if (!input_type) return failure();
    auto fft_len = rfft_op.getFftLength();
    auto fft_len_type = fft_len.getType().dyn_cast_or_null<ShapedType>();
    if (!fft_len_type) return failure();

    auto output_type =
        rfft_op.getResult().getType().dyn_cast_or_null<RankedTensorType>();
    if (!output_type) return failure();

    // Expanded inputs.
    // Insert at -2 location.
    auto one_ele_type =
        tensorflow::GetTypeFromTFTensorShape({1}, rewriter.getIntegerType(32));
    auto minus_two = CreateConstOpWithSingleValue(&rewriter, rfft_op.getLoc(),
                                                  one_ele_type, -2);

    SmallVector<int64_t, 4> expanded_input_shape;
    SmallVector<int64_t, 4> expanded_output_shape;
    int expanded_rank = input_type.getRank() + 1;
    int r = 0;
    for (int i = 0; i < expanded_rank; ++i) {
      if (i == expanded_rank - 2) {
        expanded_input_shape.push_back(1);
        expanded_output_shape.push_back(1);
      } else {
        expanded_input_shape.push_back(input_type.getDimSize(r));
        expanded_output_shape.push_back(output_type.getDimSize(r));
        r++;
      }
    }

    auto expaned_input_type = tensorflow::GetTypeFromTFTensorShape(
        expanded_input_shape, input_type.getElementType());
    TF::ExpandDimsOp expanded_input = rewriter.create<TF::ExpandDimsOp>(
        rfft_op.getLoc(), expaned_input_type, input, minus_two->getResult());

    // Expanded fft_len.
    auto one_attr = mlir::DenseIntElementsAttr::get(one_ele_type, {1});

    auto one = rewriter.create<TF::ConstOp>(rfft_op.getLoc(), one_attr);

    auto zero = CreateConstOpWithSingleValue(&rewriter, rfft_op.getLoc(),
                                             one_ele_type, 0);

    auto expanded_fft_len_type = tensorflow::GetTypeFromTFTensorShape(
        {2}, fft_len_type.getElementType());

    TF::ConcatV2Op expanded_fft_len = rewriter.create<TF::ConcatV2Op>(
        rfft_op.getLoc(), expanded_fft_len_type,
        SmallVector<Value, 2>({one.getResult(), fft_len}), zero->getResult());

    // Insert the rfft_2d.
    auto rfft2d_out_type = tensorflow::GetTypeFromTFTensorShape(
        expanded_output_shape, output_type.getElementType());
    TF::RFFT2DOp rfft2d = rewriter.create<TF::RFFT2DOp>(
        rfft_op.getLoc(), rfft2d_out_type, expanded_input.getResult(),
        expanded_fft_len.getResult());

    // Insert the squeeze op.
    auto squeeze_dim = rewriter.getI64ArrayAttr({-2});
    TF::SqueezeOp squeeze = rewriter.create<TF::SqueezeOp>(
        rfft_op.getLoc(), output_type, rfft2d.getResult(), squeeze_dim);

    rewriter.replaceOp(op, squeeze.getResult());

    return success();
  }
};

// Replaces the Identity op with its input in either of the following scenarios
// : 1) The Identity op's input and output have same types/shapes. 2) The result
// of Identity op is only used by TF ops.
struct RemoveIdentity : public OpRewritePattern<TF::IdentityOp> {
  using OpRewritePattern<TF::IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::IdentityOp identity,
                                PatternRewriter &rewriter) const override {
    // Replace the op with the input if input and result have the same type.
    if (identity.getInput().getType() == identity.getType()) {
      rewriter.replaceOp(identity, identity.getInput());
      return success();
    }
    // Replace the op with the input if output is only used by TF ops.
    // Currently this is more on the conservative side since we need to ensure
    // every consumer op to be a TF op before applying this pattern. We can
    // consider to revisit this in the future if this turns out to be too
    // restrictive.
    for (Operation *user : identity->getUsers()) {
      if (user->getDialect()->getNamespace() != "tf") {
        return failure();
      }
    }

    rewriter.replaceOp(identity, identity.getInput());
    return success();
  }
};

void PrepareTFPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  RewritePatternSet phase_2_patterns(ctx);
  auto func = getOperation();

  // Check illegal ops in a TFLite pipeline (e.g. trainning only ops) , since
  // PrepareTFPass is the very first TFLite pass in the pipeline.
  // TODO(jingpu): It might be better to split this check into its own pass
  // to make things more modular.
  if (failed(ValidateOp(func))) {
    func.emitError() << "tfl-prepare-tf pass failed.";
    signalPassFailure();
    return;
  }

  if (failed(ConvertTf2XlaOps(func, ctx))) {
    signalPassFailure();
    return;
  }

  // This pattern will try to identify and optimize for dilated convolution.
  // e.g. Patterns like "SpaceToBatchND -> Conv2D -> BatchToSpaceND" will be
  // replaced with a single Conv op with dilation parameter.
  patterns.add<ConvertTFDilatedConvOp<TF::Conv2DOp>, FusedBatchNormV3Pat,
               ConvertTFDilatedConvOp<TF::DepthwiseConv2dNativeOp>>(ctx);

  patterns.add<RemoveIdentity>(ctx);
  TFL::populateWithGenerated(patterns);
  // TODO(fengliuai): Implement similar rule in the QuantizePass if the constant
  // folding hook of tfl.transpose and tfl.reshape are implemented.
  patterns.add<ReorderFakeQuantPattern<TF::ReshapeOp>,
               ReorderFakeQuantPattern<TF::TransposeOp>>(ctx);
  // Remove redundant reshape ops.
  TF::ReshapeOp::getCanonicalizationPatterns(patterns, ctx);
  // TODO(karimnosseir): Split to separate pass probably after
  // deciding on long term plan for this optimization.
  // This will allow optimizing any TF_Mul->TF_Conv in the graph
  // and any expanded from FusedBatchNorm. We need to do this
  // before converting TF_Conv to TFL_Conv
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Remove the wrapper of the tf.FakeQuant* ops and also insert the
  // tfl.quantize and tfl.dequantize to preserve the quantization parameters.
  // This is done after the first round of optimization to make sure all the
  // min/max operands of the tf.FakeQuant* are constants to be matched. The
  // following round of optimization will folding the unwrapped
  // tf.FakeQuant* ops with the weight constants.
  if (failed(ConvertFakeQuantOps(func, ctx, use_fake_quant_num_bits_))) {
    signalPassFailure();
    return;
  }

  // Load the generated pattern again, so new quantization pass-through
  // will be applied.
  TFL::populateWithGenerated(phase_2_patterns);
  // TODO(fengliuai): Implement similar rule in the QuantizePass if the constant
  // folding hook of tfl.transpose and tfl.reshape are implemented.
  phase_2_patterns.add<ReorderFakeQuantPattern<TF::ReshapeOp>,
                       ReorderFakeQuantPattern<TF::TransposeOp>>(ctx);
  if (unfold_batch_matmul_) {
    TF::PopulateUnrollTfBatchMatMul(ctx, phase_2_patterns);
  }
  phase_2_patterns.add<TF::ConvertTFEinsumOp, ConvertTFStridedSlice,
                       ConvertRfftToRfft2d, RemoveIdentity>(ctx);
  phase_2_patterns.add<ConvertTFConv2D, ConvertTFDepthwiseConv2dNative>(
      ctx, allow_bf16_and_f16_type_legalization_);
  // Remove redundant reshape ops.
  TF::ReshapeOp::getCanonicalizationPatterns(phase_2_patterns, ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(phase_2_patterns));
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PrepareTF pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareTFPass(
    bool unfold_batch_matmul, bool allow_bf16_and_f16_type_legalization,
    bool use_fake_quant_num_bits) {
  return std::make_unique<PrepareTFPass>(unfold_batch_matmul,
                                         allow_bf16_and_f16_type_legalization,
                                         use_fake_quant_num_bits);
}

// Creates an instance of the TensorFlow Lite dialect PrepareTF pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareTFPass() {
  return std::make_unique<PrepareTFPass>();
}

}  // namespace TFL
}  // namespace mlir
