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

#include <climits>
#include <cstdint>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Optimize Pass.
namespace {

using ::llvm::cast;

// Optimize TFLite operations in functions.
struct Optimize : public FunctionPass<Optimize> {
  void runOnFunction() override;
};

// Returns whether the given type `a` is broadcast-compatible with `b`.
bool IsBroadcastableElementsAttrAndType(Type a, Type b) {
  return OpTrait::util::getBroadcastedType(a, b) != Type();
}

// Returns whether the given `a` and `b` ElementsAttr have broadcast-compatible
// types.
bool IsBroadcastableElementsAttrs(Attribute a, Attribute b) {
  return IsBroadcastableElementsAttrAndType(a.getType(), b.getType());
}

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize.inc"

// Fuse Add with proceeding FullyConnected.
// Note that this assumes that the bias in the fullyConnected
// is always None.
// TODO(b/136285429): Move to tablegen when variadic is supported
// and add support for bias with noneType type.
struct FuseFullyConnectedAndAdd : public RewritePattern {
  explicit FuseFullyConnectedAndAdd(MLIRContext *context)
      : RewritePattern(TFL::AddOp::getOperationName(),
                       {"tfl.fully_connected", "tfl.add", "std.constant"}, 4,
                       context) {}

  PatternMatchResult matchAndRewrite(Operation *add_op,
                                     PatternRewriter &rewriter) const override {
    // Fully Connected.
    Operation *fully_connected = add_op->getOperand(0)->getDefiningOp();
    if (!fully_connected || !isa<TFL::FullyConnectedOp>(fully_connected))
      return matchFailure();
    TFL::FullyConnectedOp fully_connected_op =
        llvm::cast<TFL::FullyConnectedOp>(fully_connected);
    Value *input = fully_connected_op.input();
    Value *filter = fully_connected_op.filter();

    // Make sure the bias is None.
    // TODO(karimnosseir): Support non None case.
    Operation *bias_op = fully_connected_op.bias()->getDefiningOp();
    if (!bias_op || !isa<ConstantOp>(bias_op)) return matchFailure();
    if (!fully_connected_op.bias()->getType().isa<NoneType>())
      return matchFailure();

    auto activation_func = fully_connected_op.getAttrOfType<StringAttr>(
        "fused_activation_function");
    if (!activation_func) return matchFailure();
    if (activation_func.cast<StringAttr>().getValue() != "NONE")
      return matchFailure();

    auto weight_format =
        fully_connected_op.getAttrOfType<StringAttr>("weights_format");
    if (!weight_format) return matchFailure();

    auto keep_num_dims =
        fully_connected_op.getAttrOfType<BoolAttr>("keep_num_dims");
    if (!keep_num_dims) return matchFailure();

    auto constant_op = add_op->getOperand(1)->getDefiningOp();
    if (!constant_op) return matchFailure();
    if (!isa<ConstantOp>(constant_op)) return matchFailure();

    auto add_value = constant_op->getAttrOfType<Attribute>("value");
    if (!add_value) return matchFailure();
    if (!((add_value.cast<ElementsAttr>().getType().getElementType().isF32())))
      return matchFailure();

    auto fused_activation_func =
        add_op->getAttrOfType<StringAttr>("fused_activation_function");
    if (!fused_activation_func) return matchFailure();

    // Rewrite
    // TODO(karimnosseir): Check what constraints needed to apply.
    // TODO(b/136171362): Check for single output consumer.
    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        add_op, add_op->getResult(0)->getType(),
        /*input=*/input,
        /*filter=*/filter,
        /*bias=*/add_op->getOperand(1),
        /*fused_activation_function=*/fused_activation_func,
        /*weights_format=*/weight_format,
        /*keep_num_dims=*/keep_num_dims);

    return matchSuccess();
  }
};

// TODO(b/136285429): Move to tablegen when variadic is supported.
struct FuseFullyConnectedAndRelu : public RewritePattern {
  explicit FuseFullyConnectedAndRelu(MLIRContext *context)
      : RewritePattern(TFL::ReluOp::getOperationName(), {"tfl.fully_connected"},
                       4, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto relu_op = cast<ReluOp>(op);
    Operation *input = relu_op.getOperand()->getDefiningOp();
    if (!isa_and_nonnull<FullyConnectedOp>(input)) return matchFailure();
    auto fully_connected_op = cast<FullyConnectedOp>(input);
    if (fully_connected_op.fused_activation_function() != "NONE")
      return matchFailure();

    auto new_activation_func = rewriter.getStringAttr("RELU");
    auto new_weights_format =
        rewriter.getStringAttr(fully_connected_op.weights_format());
    auto new_keep_num_dims =
        rewriter.getBoolAttr(fully_connected_op.keep_num_dims());
    rewriter.replaceOpWithNewOp<FullyConnectedOp>(
        relu_op, relu_op.getType(), fully_connected_op.input(),
        fully_connected_op.filter(), fully_connected_op.bias(),
        new_activation_func, new_weights_format, new_keep_num_dims);

    return matchSuccess();
  }
};

// Fuse Mul with proceeding FullyConnected.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseFullyConnectedAndMul : public RewritePattern {
  explicit FuseFullyConnectedAndMul(MLIRContext *context)
      : RewritePattern(TFL::MulOp::getOperationName(),
                       {"tfl.fully_connected", "tfl.mul", "std.constant"}, 4,
                       context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    // Mul.
    auto mul_op = cast<MulOp>(op);
    DenseElementsAttr cst;
    Value *constant_val = mul_op.rhs();
    if (!matchPattern(constant_val, m_Constant(&cst))) {
      return matchFailure();
    }

    // Fully Connected.
    auto fc_op =
        dyn_cast_or_null<TFL::FullyConnectedOp>(mul_op.lhs()->getDefiningOp());
    if (!fc_op) return matchFailure();
    Value *filter = fc_op.filter();
    Value *bias = fc_op.bias();
    if (fc_op.fused_activation_function().equals("None")) return matchFailure();

    // Broadcast the constant operand of Mul if it isn't compatible to the
    // filter input. We only support broadcasting the operand along the depth
    // dimension, when the operand's depth is 1.
    Value *new_const_val = constant_val;
    if (!IsBroadcastableElementsAttrAndType(cst.getType(), filter->getType())) {
      auto original_shape = cst.getType().getShape();
      llvm::SmallVector<int64_t, 4> normalized_shape(original_shape.begin(),
                                                     original_shape.end());
      normalized_shape.push_back(1);
      auto new_cst = cst.reshape(rewriter.getTensorType(
          normalized_shape, cst.getType().getElementType()));
      Type new_type = new_cst.getType();
      if (!IsBroadcastableElementsAttrAndType(new_type, filter->getType())) {
        return matchFailure();
      }
      auto new_op =
          rewriter.create<ConstantOp>(mul_op.getLoc(), new_type, new_cst);
      new_const_val = new_op.getResult();
    }

    // Rewrite.
    Location loc = fc_op.getLoc();
    auto af_none = rewriter.getStringAttr(fc_op.fused_activation_function());
    auto new_filter =
        rewriter.create<MulOp>(loc, filter, new_const_val, af_none);
    // If bias isn't None, it needs to be multiplied as well.
    if (!bias->getType().isa<NoneType>()) {
      bias = rewriter.create<MulOp>(loc, bias, constant_val, af_none).output();
    }

    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        mul_op, mul_op.getType(),
        /*input=*/fc_op.input(),
        /*filter=*/new_filter.output(),
        /*bias=*/bias,
        /*fused_activation_function=*/
        rewriter.getStringAttr(mul_op.fused_activation_function()),
        /*weights_format=*/rewriter.getStringAttr(fc_op.weights_format()),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.keep_num_dims()));

    return matchSuccess();
  }
};

// StridedSlice can have complicated atributes like begin_axis_mask,
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
struct PadStridedSliceDims : public RewritePattern {
  explicit PadStridedSliceDims(MLIRContext *context)
      : RewritePattern(TFL::StridedSliceOp::getOperationName(),
                       {"tfl.strided_slice", "tfl.strided_slice"}, 2, context) {
  }

  PatternMatchResult matchAndRewrite(Operation *strided_slice_op,
                                     PatternRewriter &rewriter) const override {
    // TODO(renjieliu): Consider expand the transformation for ellipsis & shrink
    // mask as well.
    TFL::StridedSliceOp strided_slice =
        llvm::cast<TFL::StridedSliceOp>(strided_slice_op);
    const uint64_t new_axis_mask = strided_slice.new_axis_mask().getZExtValue();
    if (new_axis_mask == 0) return matchFailure();

    // Insert a new reshape op.
    Value *original_input = strided_slice.input();
    const RankedTensorType &original_input_type =
        original_input->getType().template cast<RankedTensorType>();
    const ArrayRef<int64_t> &original_input_shape =
        original_input_type.getShape();
    const RankedTensorType &begin_type =
        strided_slice.begin()->getType().template cast<RankedTensorType>();
    const int dim_size = begin_type.getShape()[0];
    SmallVector<int64_t, 4> new_shape;
    int mask = 1;
    int index = 0;
    for (int i = 0; i < dim_size; ++i) {
      if (mask & new_axis_mask) {
        new_shape.emplace_back(1);
      } else {
        new_shape.emplace_back(original_input_shape[index]);
        ++index;
      }
      mask = mask << 1;
    }

    auto new_output_type =
        rewriter.getTensorType(new_shape, original_input_type.getElementType());

    TFL::ReshapeOp reshape = rewriter.create<TFL::ReshapeOp>(
        strided_slice.getLoc(), new_output_type, original_input);

    // Replace the original strided_slice.
    llvm::APInt new_begin_mask = strided_slice.begin_mask();
    llvm::APInt new_end_mask = strided_slice.end_mask();
    // Since we expand the dims, we need to apply them to the begin_mask &
    // end_mask.
    new_begin_mask |= strided_slice.new_axis_mask();
    new_end_mask |= strided_slice.new_axis_mask();

    auto attribute_type = rewriter.getIntegerType(32);
    rewriter.replaceOpWithNewOp<TFL::StridedSliceOp>(
        strided_slice_op, strided_slice.getType(), reshape,
        strided_slice.begin(), strided_slice.end(), strided_slice.strides(),
        rewriter.getIntegerAttr(attribute_type, new_begin_mask),
        rewriter.getIntegerAttr(attribute_type, new_end_mask),
        rewriter.getIntegerAttr(attribute_type, strided_slice.ellipsis_mask()),
        rewriter.getI32IntegerAttr(0),
        rewriter.getIntegerAttr(attribute_type,
                                strided_slice.shrink_axis_mask()));
    return matchSuccess();
  }
};

void Optimize::runOnFunction() {
  OwningRewritePatternList patterns;
  auto *ctx = &getContext();
  auto func = getFunction();

  // Add the generated patterns to the list.
  TFL::populateWithGenerated(ctx, &patterns);
  patterns.insert<FuseFullyConnectedAndAdd, FuseFullyConnectedAndRelu,
                  FuseFullyConnectedAndMul, PadStridedSliceDims>(ctx);
  applyPatternsGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
FunctionPassBase *CreateOptimizePass() { return new Optimize(); }

static PassRegistration<Optimize> pass(
    "tfl-optimize", "Optimize within the TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
