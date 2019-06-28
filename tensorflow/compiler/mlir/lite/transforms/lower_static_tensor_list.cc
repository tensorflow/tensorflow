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
// converting Tensorlist operations in TensorFlow dialect into operations that
// can be legalized to TensorFlow Lite dialect with simple replacements.  The
// newly created operations are in the TensorFlow dialect if the operation can
// be represented using a TensorFlow op.  Otherwise, TensorFlow Lite dialect op
// is used.

#include <climits>
#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/LoopAnalysis.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#define DEBUG_TYPE "tf-tfl-legalization"

//===----------------------------------------------------------------------===//
// The actual LowerStaticTensorList Pass.
//
namespace mlir {
namespace {

// Lower TensorList ops in functions for subsequent legalization.
struct LowerStaticTensorListPass
    : public FunctionPass<LowerStaticTensorListPass> {
  void runOnFunction() override;
  LogicalResult ModifyTensorList();
};

Value *CreateI32SplatConst(Operation *op, PatternRewriter *rewriter,
                           ArrayRef<int64_t> shape, int32_t val) {
  auto type = rewriter->getTensorType(shape, rewriter->getIntegerType(32));
  auto attr = DenseElementsAttr::get(type, rewriter->getI32IntegerAttr(val));
  return rewriter->create<ConstantOp>(op->getLoc(), type, attr);
}

Value *CreateI32SplatTensor(Operation *op, PatternRewriter *rewriter,
                            Value *shape_tensor, int32_t val) {
  auto scalar_val = CreateI32SplatConst(op, rewriter, {}, val);
  return rewriter->create<TF::FillOp>(
      op->getLoc(), rewriter->getTensorType({-1}, rewriter->getIntegerType(32)),
      shape_tensor, scalar_val);
}

struct ConvertTFTensorListSetItem : public RewritePattern {
  explicit ConvertTFTensorListSetItem(MLIRContext *context)
      : RewritePattern(TF::TensorListSetItemOp::getOperationName(), 1,
                       context) {}
  // This function rewrites the original op into a series of slice and concat op
  // to produce the same result. It first slices the first `$index` rows. Then
  // expands the dimension of the `$item`, followed by another slice of the
  // remaining rows starting from `$index` + 1. Lastly it concatenates the
  // three parts together.
  // On a high level, it's doing something like:
  // def : Pat<(TF_TensorListSetItemOp $input, $index, $item),
  //      (Concat
  //        concat_dim = 0,
  //        (Slice $input, [0, 0, ...], (Concat (ExpandDims $index, expand_dim =
  //        0), [-1, -1, ...])), (ExpandDims $item, expand_dim = 0), (Slice
  //        $input, [$index + 1, 0, 0, ...], [-1, -1, ...]))>;
  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TF::TensorListSetItemOp tf_op = cast<TF::TensorListSetItemOp>(op);

    auto input = tf_op.input_handle();
    auto shape_dtype = rewriter.getIntegerType(32);
    auto input_rank = rewriter.create<TF::RankOp>(
        op->getLoc(), rewriter.getTensorType({}, shape_dtype), input);
    auto item = tf_op.item();
    auto item_rank = rewriter.create<TF::RankOp>(
        op->getLoc(), rewriter.getTensorType({}, shape_dtype), item);

    // Prepare the start position for the first slice op, which is [0, 0, ..,
    // 0].
    auto scalar_zero = CreateI32SplatConst(op, &rewriter, {}, 0);
    auto position_shape = rewriter.create<TF::ExpandDimsOp>(
        op->getLoc(), rewriter.getTensorType({1}, shape_dtype), input_rank,
        scalar_zero);
    // Fill all 0s into the first position tensor.
    auto first_start_position =
        CreateI32SplatTensor(op, &rewriter, position_shape, 0);

    // Prepare the start position for the second slice op, which is
    // [index + 1, 0, 0 .. 0].
    // Calculate the first dimension, which is index + 1.
    auto index = tf_op.index();
    auto vector_type = rewriter.getTensorType({1}, shape_dtype);
    auto begin =
        rewriter.create<TF::AddOp>(op->getLoc(), vector_type, index,
                                   CreateI32SplatConst(op, &rewriter, {1}, 1));

    // Followed by the first dimension `begin`, are `item_rank` of 0s.
    auto item_position_shape = rewriter.create<TF::ExpandDimsOp>(
        op->getLoc(), rewriter.getTensorType({1}, shape_dtype), item_rank,
        scalar_zero);
    auto partial_second_start_position =
        CreateI32SplatTensor(op, &rewriter, item_position_shape, 0);
    auto position_type = first_start_position->getType();
    // Concatenate `begin` with the remaining 0s.
    auto second_start_position = rewriter.create<TF::ConcatOp>(
        op->getLoc(), position_type, scalar_zero,
        ArrayRef<Value *>({begin, partial_second_start_position}),
        rewriter.getI64IntegerAttr(2));

    // Create the size parameter for the first slice op, which is [index, -1,
    // -1, .., -1].
    auto size1_leading_dim = rewriter.create<TF::ExpandDimsOp>(
        op->getLoc(), vector_type, index, scalar_zero);
    auto partial_size1 =
        CreateI32SplatTensor(op, &rewriter, item_position_shape, -1);
    auto size1 = rewriter.create<TF::ConcatOp>(
        op->getLoc(), position_type, scalar_zero,
        ArrayRef<Value *>({size1_leading_dim, partial_size1}),
        rewriter.getI64IntegerAttr(2));

    // Create the size parameter for the second slice, which is [-1, -1, ..,
    // -1].
    auto size2 = CreateI32SplatTensor(op, &rewriter, position_shape, -1);

    // Create two slice ops.
    auto element_type = input->getType().cast<TensorType>().getElementType();
    auto unranked_tensor = rewriter.getTensorType(element_type);
    auto slice1 = rewriter.create<TF::SliceOp>(
        op->getLoc(), unranked_tensor, input, first_start_position, size1);
    auto slice2 = rewriter.create<TF::SliceOp>(
        op->getLoc(), unranked_tensor, input, second_start_position, size2);

    // Expand the dimension of item so that it will have the same rank with
    // input.
    auto expanded_item = rewriter.create<TF::ExpandDimsOp>(
        op->getLoc(), unranked_tensor, item, scalar_zero);

    // Concatenate three parts together to generate the final result.
    rewriter.replaceOpWithNewOp<TF::ConcatOp>(
        op, input->getType(), scalar_zero,
        ArrayRef<Value *>({slice1, expanded_item, slice2}),
        rewriter.getI64IntegerAttr(3));

    return matchSuccess();
  }
};

// TODO(hinsu): Fix end-to-end test when passing string `element_dtype`
// attribute.
struct ConvertTFTensorListReserve : public RewritePattern {
  explicit ConvertTFTensorListReserve(MLIRContext *context)
      : RewritePattern(TF::TensorListReserveOp::getOperationName(), 1,
                       context) {}

  // Rewrites the original op into `tf.fill`. The result tensor shape is
  // [num_element, element_shape]. All the values in the result tensor will be
  // initialized to 0.
  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TF::TensorListReserveOp tf_op = cast<TF::TensorListReserveOp>(op);

    auto element_shape = tf_op.element_shape();
    auto shape_dtype =
        element_shape->getType().cast<TensorType>().getElementType();
    auto num_elements = tf_op.num_elements();
    int64_t input_rank = -1;  // -1 means unknown dimension.
    if (auto type = element_shape->getType().dyn_cast<RankedTensorType>()) {
      // Note that the first item of the shape array is the element's rank, add
      // it by 1 to get the input's rank.
      if (type.hasStaticShape()) {
        input_rank = type.getShape()[0] + 1;
      }
    }
    auto element_dtype = tf_op.element_dtype();

    // The output shape of the result tensor should be [num_elements +
    // element_shape].
    auto scalar_zero = CreateI32SplatConst(op, &rewriter, {}, 0);
    auto leading_dim = rewriter.create<TF::ExpandDimsOp>(
        op->getLoc(), rewriter.getTensorType({1}, shape_dtype), num_elements,
        scalar_zero);
    auto shape_type = rewriter.getTensorType({input_rank}, shape_dtype);
    auto list_shape = rewriter.create<TF::ConcatOp>(
        op->getLoc(), shape_type, scalar_zero,
        ArrayRef<Value *>({leading_dim, element_shape}),
        rewriter.getI64IntegerAttr(2));

    // Create a zero-initialized constant tensor that has the same type
    // as specified by element_dtype.
    auto zero_type = rewriter.getTensorType({}, element_dtype);
    auto zero_attr = rewriter.getZeroAttr(zero_type);
    auto zero = rewriter.create<ConstantOp>(op->getLoc(), zero_type, zero_attr);

    rewriter.replaceOpWithNewOp<TF::FillOp>(
        op, rewriter.getTensorType(element_dtype), list_shape, zero);
    return matchSuccess();
  }
};

}  // namespace

namespace TFL {
namespace {
#include "tensorflow/compiler/mlir/lite/transforms/generated_lower_static_tensor_list.inc"
}  // namespace
}  // namespace TFL

LogicalResult LowerStaticTensorListPass::ModifyTensorList() {
  // In `runOnFunction`, there is no guarantee about
  // in which order those patterns will be applied. Our transformation requires
  // that at runtime each `TensorListSetItem` op takes in a normal tensor type
  // rather than a `DT_VARIANT` tensor. So here we need to manually walk-through
  // the IR and change the argument/return types of each `TensorListSetItemOp`.
  // TODO(haoliang): 1) support modifying more `TensorList` ops that consumes/
  // produces `DT_VARIANT` tensor. 2) More robust support for handling multiple
  // different tensorlist types. For example, consider the case like:
  // l1 = list_ops.tensor_list_from_tensor(t, element_shape1)
  // l2 = list_ops.tensor_list_from_tensor(t, element_shape2)
  // l1 = list_ops.tensor_list_set_item(l1, 0, item1)
  // l2 = list_ops.tensor_list_set_item(l2, 0, item2)
  // 3) Handle the case where a tensorlist output is passed to multiple
  // functions.
  for (Block &block : getFunction()) {
    Type tensor_type;
    for (Operation &op : block) {
      if (auto tf_op = llvm::dyn_cast<TF::TensorListFromTensorOp>(op)) {
        tensor_type = tf_op.tensor()->getType();
      } else if (auto tf_op = llvm::dyn_cast<TF::TensorListReserveOp>(op)) {
        if (!(tf_op.element_dtype().isF16() || tf_op.element_dtype().isF32() ||
              tf_op.element_dtype().isF64() ||
              tf_op.element_dtype().isa<IntegerType>())) {
          return tf_op.emitError(
              "requires element_dtype to be integer or 16-bit/32-bit/64-bit "
              "float type during TF Lite transformation pass");
        }
        // TODO(haoliang): figure out better way of specify shape.
        tensor_type = UnrankedTensorType::get(tf_op.element_dtype());
      }

      if (auto tf_op = llvm::dyn_cast<TF::TensorListSetItemOp>(op)) {
        tf_op.input_handle()->setType(tensor_type);
        tf_op.getResult()->setType(tensor_type);
      }
      // Currently we will raise an error if an op other than the following
      // contains a DT_VARIANT tensor as its input or output. Below ops already
      // have proper transformation patterns that eliminate the need of
      // `DT_VARIANT`, we consider it's safe to not raise an error on those ops.
      if (llvm::isa<TF::TensorListFromTensorOp>(op) ||
          llvm::isa<TF::TensorListReserveOp>(op) ||
          llvm::isa<TF::TensorListSetItemOp>(op) ||
          llvm::isa<TF::TensorListStackOp>(op) ||
          llvm::isa<TF::TensorListGetItemOp>(op)) {
        continue;
      }
      // Check if any of the input operand is a DT_VARIANT.
      for (Type type : op.getOperandTypes()) {
        if (type.isa<TF::VariantType>()) {
          return op.emitError(
              "op's input contains a DT_VARIANT tensor. Currently we only "
              "allow "
              "TensorListFromTensor/TensorListReserve/TensorListStack/"
              "TensorListSetItem/"
              "TensorListGetItem to have DT_VARIANT input/output");
        }
      }
      // Check if any of the output is a DT_VARIANT.
      for (Type type : op.getResultTypes()) {
        if (type.isa<TF::VariantType>()) {
          return op.emitError(
              "op's output contains a DT_VARIANT tensor. Currently we only "
              "allow "
              "TensorListFromTensor/TensorListReserve/TensorListStack/"
              "TensorListSetItem/"
              "TensorListGetItem to have DT_VARIANT input/output");
        }
      }
    }
  }
  return success();
}

void LowerStaticTensorListPass::runOnFunction() {
  if (failed(ModifyTensorList())) {
    signalPassFailure();
    return;
  }
  OwningRewritePatternList patterns;
  auto &func = getFunction();
  TFL::populateWithGenerated(&getContext(), &patterns);
  patterns.push_back(
      llvm::make_unique<ConvertTFTensorListReserve>(&getContext()));
  patterns.push_back(
      llvm::make_unique<ConvertTFTensorListSetItem>(&getContext()));
  applyPatternsGreedily(func, std::move(patterns));
}

// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
// pass.
FunctionPassBase *TFL::CreateLowerStaticTensorListPass() {
  return new LowerStaticTensorListPass();
}

static PassRegistration<LowerStaticTensorListPass> pass(
    "tfl-lower-static-tensor-list",
    "Lower TensorList ops within TensorFlow Lite dialect");

}  // namespace mlir
