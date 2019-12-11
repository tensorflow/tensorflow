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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/LoopAnalysis.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/SymbolTable.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "mlir/Transforms/DialectConversion.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#define DEBUG_TYPE "tf-tfl-legalization"

//===----------------------------------------------------------------------===//
// The actual LowerStaticTensorList Pass.
//
namespace mlir {
namespace {

class TensorListPatternRewriter : public PatternRewriter {
 public:
  explicit TensorListPatternRewriter(FuncOp fn)
      : PatternRewriter(fn.getContext()) {}

  Operation *createOperation(const OperationState &state) override {
    return OpBuilder::createOperation(state);
  }
};

/// Lower TensorList ops in functions for subsequent legalization.
struct LowerStaticTensorListPass
    : public ModulePass<LowerStaticTensorListPass> {
  void runOnModule() override;

  // Apply type and op changes within a function.
  LogicalResult RewriteFunction(FuncOp func,
                                TensorListPatternRewriter *rewriter);
};

Value *CreateI32SplatConst(Location loc, PatternRewriter *rewriter,
                           ArrayRef<int64_t> shape, int32_t val) {
  RankedTensorType type =
      RankedTensorType::get(shape, rewriter->getIntegerType(32));
  DenseElementsAttr attr =
      DenseElementsAttr::get(type, rewriter->getI32IntegerAttr(val));
  return rewriter->create<ConstantOp>(loc, type, attr);
}

Value *CreateI32SplatTensor(Location loc, PatternRewriter *rewriter,
                            Value *shape_tensor, int32_t val) {
  Value *scalar_val = CreateI32SplatConst(loc, rewriter, {}, val);
  return rewriter->create<TF::FillOp>(
      loc, RankedTensorType::get({-1}, rewriter->getIntegerType(32)),
      shape_tensor, scalar_val);
}

// Returns a new type by prepending the specified dimension to the shape of
// the given type if it is a ranked type.
Type PrependLeadingDimIfRanked(int64_t dim, Type type,
                               PatternRewriter *rewriter) {
  Type dtype = getElementTypeOrSelf(type);
  if (RankedTensorType ty = type.dyn_cast<RankedTensorType>()) {
    llvm::SmallVector<int64_t, 4> shape = {dim};
    shape.append(ty.getShape().begin(), ty.getShape().end());
    return RankedTensorType::get(shape, dtype);
  }
  return type;
}

Type GetTensorTypeForTensorList(Type element_type, TF::VariantType handle_dtype,
                                PatternRewriter *rewriter) {
  // If the variant type in the output handle has item shape available, use it
  // to derive the output shape by setting unknown leading dimension.
  // Otherwise, result type will be of unranked type.
  if (handle_dtype.getSubtypes().empty()) {
    return UnrankedTensorType::get(element_type);
  }
  return PrependLeadingDimIfRanked(-1, handle_dtype.getSubtypes()[0], rewriter);
}

// Creates a slice of the tensorlist `input_list`, starting from
// [start_index, 0, ...0], with size [size, -1, ...-1].
//
// Requires that `start_index` and `size` are scalar tensors and
// `item_position_shape` is a 1-D tensor with only one element equal to the rank
// of an item in the tensorlist.
TF::SliceOp CreateSliceOpForTensorList(Location loc, Value *input_list,
                                       Value *start_index, Value *size,
                                       Value *item_rank, Type result_type,
                                       PatternRewriter *rewriter) {
  // Create the start position of slice. This is done by concatenating
  // `start_index` and `partial_start_position` together.
  IntegerType shape_dtype = rewriter->getIntegerType(32);
  RankedTensorType position_type = RankedTensorType::get({-1}, shape_dtype);
  Value *partial_start_position =
      CreateI32SplatTensor(loc, rewriter, item_rank, 0);
  Value *scalar_zero = CreateI32SplatConst(loc, rewriter, {}, 0);
  RankedTensorType vector_type = RankedTensorType::get({1}, shape_dtype);
  auto expanded_start_index = rewriter->create<TF::ExpandDimsOp>(
      loc, vector_type, start_index, scalar_zero);
  auto start_position = rewriter->create<TF::ConcatOp>(
      loc, position_type, scalar_zero,
      ArrayRef<Value *>({expanded_start_index, partial_start_position}));

  // Create the slice size tensor. This is done by concatenating `size` and
  // `partial_size`.
  auto size_leading_dim =
      rewriter->create<TF::ExpandDimsOp>(loc, vector_type, size, scalar_zero);
  Value *partial_size = CreateI32SplatTensor(loc, rewriter, item_rank, -1);
  auto slice_size = rewriter->create<TF::ConcatOp>(
      loc, position_type, scalar_zero,
      ArrayRef<Value *>({size_leading_dim, partial_size}));

  return rewriter->create<TF::SliceOp>(loc, result_type, input_list,
                                       start_position, slice_size);
}

struct ConvertTensorListSetItem : public ConversionPattern {
  explicit ConvertTensorListSetItem(MLIRContext *context)
      : ConversionPattern(TF::TensorListSetItemOp::getOperationName(), 1,
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
  PatternMatchResult matchAndRewrite(
      Operation *operation, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto op = llvm::cast<TF::TensorListSetItemOp>(operation);
    Location loc = op.getLoc();
    Value *input = operands[0];
    Value *index = operands[1];
    Value *item = operands[2];

    IntegerType shape_dtype = rewriter.getIntegerType(32);
    auto item_rank = rewriter.create<TF::RankOp>(
        loc, RankedTensorType::get({}, shape_dtype), item);
    Value *scalar_zero = CreateI32SplatConst(loc, &rewriter, {}, 0);

    // Calculate `index` + 1, which is used to generate the start position for
    // the second slice op.
    auto suffix_start =
        rewriter.create<TF::AddOp>(loc, index->getType(), index,
                                   CreateI32SplatConst(loc, &rewriter, {}, 1));

    auto item_position_shape = rewriter.create<TF::ExpandDimsOp>(
        loc, RankedTensorType::get({1}, shape_dtype), item_rank, scalar_zero);
    // Create two slice ops.
    Type element_type = input->getType().cast<TensorType>().getElementType();
    UnrankedTensorType unranked_tensor = UnrankedTensorType::get(element_type);
    Value *scalar_minus_one = CreateI32SplatConst(loc, &rewriter, {}, -1);
    TF::SliceOp slice1 =
        CreateSliceOpForTensorList(loc, /*input_list=*/input,
                                   /*start_index=*/scalar_zero,
                                   /*size=*/index,
                                   /*item_rank=*/item_position_shape,
                                   /*result_type=*/unranked_tensor, &rewriter);
    TF::SliceOp slice2 =
        CreateSliceOpForTensorList(loc, /*input_list=*/input,
                                   /*start_index=*/suffix_start,
                                   /*size=*/scalar_minus_one,
                                   /*item_rank=*/item_position_shape,
                                   /*result_type=*/unranked_tensor, &rewriter);

    // Expand the dimension of item so that it will have the same rank with
    // input.
    auto expanded_item = rewriter.create<TF::ExpandDimsOp>(
        op.getLoc(), unranked_tensor, item, scalar_zero);

    // Concatenate three parts together to generate the final result.
    rewriter.replaceOpWithNewOp<TF::ConcatOp>(
        op, input->getType(), scalar_zero,
        ArrayRef<Value *>({slice1, expanded_item, slice2}));
    return matchSuccess();
  }
};

// Rewrites op of the template type initializing a TensorList with a list of ops
// to generate an equivalent raw tensor. Derived classes are required to
// override GetNumElements method.
template <typename OpT>
struct ConvertTensorListInitOp : public ConversionPattern {
  explicit ConvertTensorListInitOp(MLIRContext *context)
      : ConversionPattern(OpT::getOperationName(), 1, context) {}

  // Create and return a 1-d tensor with exactly one element equal to the number
  // of list elements to initialize the output tensor list with.
  virtual Value *GetNumElements(OpT op, ArrayRef<Value *> operands,
                                PatternRewriter *rewriter) const = 0;

  // Rewrites the original op into `tf.fill`. The result tensor shape is
  // [num_element, element_shape]. All the values in the result tensor will be
  // initialized to 0.
  PatternMatchResult matchAndRewrite(
      Operation *operation, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    OpT op = llvm::cast<OpT>(operation);

    Type dtype = op.element_dtype();
    if (!(dtype.isF16() || dtype.isF32() || dtype.isF64() ||
          dtype.isInteger(1) || dtype.isInteger(8) || dtype.isInteger(16) ||
          dtype.isInteger(32) || dtype.isInteger(64))) {
      op.emitError(
          "requires element_dtype to be 1-bit/8-bit/16-bit/32-bit/64-bit "
          "integer or 16-bit/32-bit/64-bit float type during TF Lite "
          "transformation pass");
      return matchFailure();
    }

    Value *element_shape = operands[0];
    Type shape_dtype = getElementTypeOrSelf(element_shape->getType());

    DenseIntElementsAttr dense_elem_attr;
    if (matchPattern(element_shape, m_Constant(&dense_elem_attr))) {
      // Note: It's technically unsafe to rewrite
      //     TensorListReserve(num_element, element_shape)
      // to
      //     Fill(Concat(num_element, element_shape), 0)
      // because element_shape may contain -1 to represent unknown dimension.
      //
      // In real world use cases (e.g. Keras RNN), `element_shape` is usually
      // a constant, and the first dimension of `element_shape` is usually
      // batch dimension. Currently TFLiteConverter always rewrite unknown
      // batch dimension to 1, therefore we also rewrite unknown dimension in
      // `element_shape` to 1 here.
      //
      // This workaround enables converting Keras RNN without specifying batch
      // dimension. This isn't guaranteed to work, but it doesn't break any
      // non-broken cases either (since it's already broken if `element_shape`
      // contains -1).
      // TODO(b/142096690): Support dynamic element shape and remove the
      // workaround.
      SmallVector<int32_t, 4> new_element_shape_values;

      auto int_values = dense_elem_attr.getIntValues();
      for (auto it = int_values.begin(); it != int_values.end(); ++it) {
        auto dim_value = (*it).getSExtValue();
        if (it == int_values.begin() && dim_value == -1) {
          dim_value = 1;
        }
        new_element_shape_values.push_back(dim_value);
      }

      auto attr =
          DenseIntElementsAttr::get(element_shape->getType().cast<ShapedType>(),
                                    new_element_shape_values);
      auto new_element_shape = rewriter.create<ConstantOp>(
          op.getLoc(), element_shape->getType(), attr);
      element_shape = new_element_shape;
    }

    int64_t result_rank = -1;  // -1 means unknown result rank.
    Type element_dtype = op.element_dtype();
    Type result_type = UnrankedTensorType::get(element_dtype);
    if (auto element_type =
            op.element_type().template dyn_cast<RankedTensorType>()) {
      result_rank = element_type.getRank() + 1;
      // If element type is ranked, then result type will have unknown leading
      // dimension and element shape for the following dimensions.
      //
      // Note: leading dim is not inferred here even when it is a constant.
      SmallVector<int64_t, 4> result_shape = {-1};
      ArrayRef<int64_t> shape = element_type.getShape();
      result_shape.append(shape.begin(), shape.end());
      result_type = RankedTensorType::get(result_shape, element_dtype);
    }

    // Create a 1-D RankedTensorType for result's shape. Number of elements in
    // it is equal to the rank of the result, if known. Otherwise, the number of
    // elements are unknown and represented with -1. In both cases, we can
    // specify dimension using rank of the result.
    Type shape_type = RankedTensorType::get({result_rank}, shape_dtype);

    Location loc = op.getLoc();
    // Add number of elements as the prefix to the element shape to get shape of
    // the output tensor.
    Value *leading_dim = GetNumElements(op, operands, &rewriter);
    Value *scalar_zero = CreateI32SplatConst(loc, &rewriter, {}, 0);
    auto list_shape = rewriter.create<TF::ConcatOp>(
        loc, shape_type, scalar_zero,
        ArrayRef<Value *>({leading_dim, element_shape}));

    // Create a zero-initialized constant tensor that has the same type
    // as specified by element_dtype.
    RankedTensorType zero_type = RankedTensorType::get({}, element_dtype);
    Attribute zero_attr = rewriter.getZeroAttr(zero_type);
    auto zero = rewriter.create<ConstantOp>(loc, zero_type, zero_attr);

    rewriter.replaceOpWithNewOp<TF::FillOp>(op, result_type, list_shape, zero);
    return Pattern::matchSuccess();
  }
};

struct ConvertTensorListReserve
    : public ConvertTensorListInitOp<TF::TensorListReserveOp> {
  explicit ConvertTensorListReserve(MLIRContext *context)
      : ConvertTensorListInitOp(context) {}

  Value *GetNumElements(TF::TensorListReserveOp op, ArrayRef<Value *> operands,
                        PatternRewriter *rewriter) const override {
    Value *scalar_zero = CreateI32SplatConst(op.getLoc(), rewriter, {}, 0);
    Type shape_dtype = getElementTypeOrSelf(op.element_shape()->getType());
    Value *num_elements = operands[1];
    return rewriter->create<TF::ExpandDimsOp>(
        op.getLoc(), RankedTensorType::get({1}, shape_dtype), num_elements,
        scalar_zero);
  }
};

// Note that we ignore the second operand `max_num_elements` as we don't have
// any restrictions on the number of elements we can support. So this may
// have a different behavior compared to TensorFlow in case of errors.
struct ConvertEmptyTensorList
    : public ConvertTensorListInitOp<TF::EmptyTensorListOp> {
  explicit ConvertEmptyTensorList(MLIRContext *context)
      : ConvertTensorListInitOp(context) {}

  Value *GetNumElements(TF::EmptyTensorListOp op, ArrayRef<Value *> operands,
                        PatternRewriter *rewriter) const override {
    return CreateI32SplatConst(op.getLoc(), rewriter, {1}, 0);
  }
};

struct ConvertTensorListPushBack : public ConversionPattern {
  explicit ConvertTensorListPushBack(MLIRContext *context)
      : ConversionPattern(TF::TensorListPushBackOp::getOperationName(), 1,
                          context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    TF::TensorListPushBackOp push_back_op = cast<TF::TensorListPushBackOp>(op);
    Value *input_handle = operands[0];
    Value *item = operands[1];

    // Expand the shape of the item so that it will have rank same as the input
    // tensor and it is compatible for the Concat Op.
    Type expanded_item_type =
        PrependLeadingDimIfRanked(1, item->getType(), &rewriter);
    Value *scalar_zero = CreateI32SplatConst(op->getLoc(), &rewriter, {}, 0);
    auto expanded_item = rewriter.create<TF::ExpandDimsOp>(
        op->getLoc(), expanded_item_type, item, scalar_zero);

    Type elem_type = getElementTypeOrSelf(item);
    auto handle_dtype =
        getElementTypeOrSelf(push_back_op.output_handle()->getType())
            .cast<TF::VariantType>();
    Type result_type =
        GetTensorTypeForTensorList(elem_type, handle_dtype, &rewriter);

    // Concatenate tensor stored in the input handle with the expanded item to
    // get a tensor equivalent to the TensorList generated by this op.
    rewriter.replaceOpWithNewOp<TF::ConcatOp>(
        push_back_op, result_type, scalar_zero,
        ArrayRef<Value *>({input_handle, expanded_item}));
    return matchSuccess();
  }
};

// Rewrites `TensorListResize` op into a functional `If` op and several basic
// TF ops to match the op semantics of Tensorflow. Basically, it does:
// 1) If the requested size is smaller or equal than the input tensorlist's
// size, rewrite it to a Slice op so that only the first 'size' rows are
// returned. 2) If the requested size is larger than the input tensorlist's
// size. We need to create an additional tensorlist with 'size - input_size'
// elements, and append it to the end of the input tensorlist.
// TODO(haoliang): We could simplify this transformation by rewriting to pure
// tensorlist ops and a few non-tensorlist ops (such as `SliceOp`). By operating
// only on variant types, we could save some ops involved in rewriting this op.
struct ConvertTensorListResize : public ConversionPattern {
  explicit ConvertTensorListResize(MLIRContext *context)
      : ConversionPattern(TF::TensorListResizeOp::getOperationName(), 1,
                          context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    TF::TensorListResizeOp resize_op = cast<TF::TensorListResizeOp>(op);
    Value *input_handle = operands[0];
    Value *size = operands[1];

    Location loc = resize_op.getLoc();
    Value *scalar_zero = CreateI32SplatConst(loc, &rewriter, {}, 0);

    // Compute the input tensorlist's length and store it in `input_size`.
    IntegerType shape_dtype = rewriter.getIntegerType(32);
    auto input_size = rewriter.create<TF::TensorListLengthOp>(
        loc, RankedTensorType::get({}, shape_dtype), op->getOperand(0));

    // Infer result type of this op based on TF's shape inference result.
    Type elem_type = getElementTypeOrSelf(input_handle);
    auto handle_dtype =
        getElementTypeOrSelf(resize_op.output_handle()->getType())
            .cast<TF::VariantType>();
    Type result_type =
        GetTensorTypeForTensorList(elem_type, handle_dtype, &rewriter);

    // Compute the difference of `size` and `input_size`, and store it in
    // `size_diff`, which is then consumed by `if_cond`.
    auto size_diff = rewriter.create<TF::SubOp>(
        loc, RankedTensorType::get({}, shape_dtype), size, input_size);
    auto if_cond = rewriter.create<TF::GreaterOp>(
        loc, RankedTensorType::get({}, rewriter.getI1Type()), size_diff,
        scalar_zero);

    // Build the argument/result types for if branch function.
    auto input_shape = rewriter.create<TF::ShapeOp>(
        loc, RankedTensorType::get({-1}, shape_dtype), input_handle);

    Type branch_args_type[] = {input_handle->getType(), input_shape.getType(),
                               size_diff.getType(), size->getType()};
    Type branch_result_type[] = {result_type};
    auto func_type = FunctionType::get(branch_args_type, branch_result_type,
                                       rewriter.getContext());

    // Constructs `then_branch`, which is executed when `if_cond` evaluates to
    // true.
    FuncOp then_branch_op = FuncOp::create(loc, "cond_true", func_type);
    CreateCondTrueBranch(resize_op, shape_dtype, result_type, then_branch_op,
                         &rewriter);

    // Constructs `else_branch`, which is executed when `if_cond` evaluates to
    // false.
    FuncOp else_branch_op = FuncOp::create(loc, "cond_false", func_type);
    CreateCondFalseBranch(loc, shape_dtype, result_type, else_branch_op,
                          &rewriter);

    // Inserts the two blocks' names into the symbol table held by the module.
    // Using SymbolTable will ensure that the inserted symbol names are
    // unique.
    SymbolTable manager(resize_op.getParentOfType<ModuleOp>());
    manager.insert(then_branch_op);
    manager.insert(else_branch_op);

    rewriter.replaceOpWithNewOp<TF::IfOp>(
        op, result_type, if_cond,
        /*input=*/
        ArrayRef<Value *>({input_handle, input_shape, size_diff, size}),
        /*then_branch=*/rewriter.getSymbolRefAttr(then_branch_op),
        /*else_branch=*/rewriter.getSymbolRefAttr(else_branch_op),
        /*output_shapes=*/rewriter.getStrArrayAttr({"{}"}),
        /*is_stateless=*/rewriter.getBoolAttr(true));
    return matchSuccess();
  }

 private:
  // When the input tensorlist's size is smaller than the requested size,
  // then branch is executed.
  // Create a new tensorlist of size 'size - input_size' and concat it
  // with the input tensorlist.
  void CreateCondTrueBranch(TF::TensorListResizeOp resize_op, Type shape_dtype,
                            Type result_type, FuncOp branch_func,
                            ConversionPatternRewriter *rewriter) const {
    auto guard = OpBuilder::InsertionGuard(*rewriter);
    Block *block = branch_func.addEntryBlock();
    rewriter->setInsertionPointToStart(block);

    auto input_shape = block->getArgument(1);
    auto size_diff = block->getArgument(2);
    auto input = block->getArgument(0);

    Location loc = resize_op.getLoc();
    // Get the element shape by slicing from index 1 in the input shape.
    Value *slice_size = CreateI32SplatConst(loc, rewriter, {1}, -1);
    Value *scalar_zero = CreateI32SplatConst(loc, rewriter, {}, 0);
    Value *slice_start = CreateI32SplatConst(loc, rewriter, {1}, 1);
    auto elem_shape = rewriter->create<TF::SliceOp>(
        loc, RankedTensorType::get({-1}, shape_dtype), input_shape, slice_start,
        slice_size);
    auto extended_part = rewriter->create<TF::TensorListReserveOp>(
        loc, resize_op.output_handle()->getType(), elem_shape, size_diff);
    // `ConcatOp` expects non-variant-typed input. Insert a
    // `TensorListStackOp` here to convert type from variant to non-variant.
    // Note that we are using the same `result_type` for both the
    // `TensorListStackOp` and `ConcatOp`, since the first dimension of the
    // shape specified by `result_type` is -1.
    auto stacked_extended_part = rewriter->create<TF::TensorListStackOp>(
        loc, result_type, extended_part,
        /*element_shape=*/CreateI32SplatConst(loc, rewriter, {}, -1),
        /*num_elements=*/rewriter->getI32IntegerAttr(-1));
    auto concat_op = rewriter->create<TF::ConcatOp>(
        loc, result_type, scalar_zero,
        ArrayRef<Value *>({input, stacked_extended_part}));
    rewriter->create<ReturnOp>(loc, ArrayRef<Value *>({concat_op}));
  }

  void CreateCondFalseBranch(Location loc, Type shape_dtype, Type result_type,
                             FuncOp branch_func,
                             ConversionPatternRewriter *rewriter) const {
    // When the input tensorlist's size is larger or equal than the requested
    // size, the else branch is executed.
    // Slice the first 'size' rows from the input tensorlist.
    auto guard = OpBuilder::InsertionGuard(*rewriter);
    Block *block = branch_func.addEntryBlock();
    rewriter->setInsertionPointToStart(block);

    Value *scalar_zero = CreateI32SplatConst(loc, rewriter, {}, 0);
    Value *vector_one = CreateI32SplatConst(loc, rewriter, {1}, 1);
    auto input = block->getArgument(0);
    auto size = block->getArgument(3);

    // Subtract `input_rank` by 1 to get the item's rank, which is used as
    // `partial_position_shape`.
    auto input_rank = rewriter->create<TF::RankOp>(
        loc, RankedTensorType::get({}, shape_dtype), input);
    auto partial_position_shape = rewriter->create<TF::SubOp>(
        loc, RankedTensorType::get({1}, shape_dtype), input_rank, vector_one);
    auto slice_op =
        CreateSliceOpForTensorList(loc, /*input_list=*/input,
                                   /*start_index=*/scalar_zero, /*size=*/size,
                                   /*item_rank=*/partial_position_shape,
                                   /*result_type=*/result_type, rewriter);
    rewriter->create<ReturnOp>(loc, ArrayRef<Value *>({slice_op}));
  }
};

struct ConvertTensorListGetItem : public ConversionPattern {
  explicit ConvertTensorListGetItem(MLIRContext *context)
      : ConversionPattern(TF::TensorListGetItemOp::getOperationName(), 1,
                          context) {}

  PatternMatchResult matchAndRewrite(
      Operation *operation, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto op = llvm::cast<TF::TensorListGetItemOp>(operation);
    Value *input = operands[0];
    Value *index = operands[1];
    rewriter.replaceOpWithNewOp<TF::GatherOp>(
        operation, op.getType(), input, index, rewriter.getBoolAttr(true));
    return matchSuccess();
  }
};

struct ConvertTensorListLength : public ConversionPattern {
  explicit ConvertTensorListLength(MLIRContext *context)
      : ConversionPattern(TF::TensorListLengthOp::getOperationName(), 1,
                          context) {}

  PatternMatchResult matchAndRewrite(
      Operation *operation, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto op = llvm::cast<TF::TensorListLengthOp>(operation);
    Location loc = op.getLoc();
    Value *input_handle = operands[0];

    BoolAttr true_attr = rewriter.getBoolAttr(true);
    auto shape = rewriter.create<TF::ShapeOp>(loc, input_handle,
                                              /*use_32bit=*/true_attr);
    rewriter.replaceOpWithNewOp<TF::GatherOp>(
        op, op.getType(), shape, CreateI32SplatConst(loc, &rewriter, {}, 0),
        /*validate_indices=*/true_attr);
    return matchSuccess();
  }
};

struct ConvertTensorListStack : public ConversionPattern {
  explicit ConvertTensorListStack(MLIRContext *context)
      : ConversionPattern(TF::TensorListStackOp::getOperationName(), 1,
                          context) {}

  PatternMatchResult matchAndRewrite(
      Operation *operation, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto op = llvm::cast<TF::TensorListStackOp>(operation);
    Location loc = op.getLoc();
    Value *input = operands[0];
    Value *element_shape = operands[1];

    // If the `element_shape` is a known constant (which is defined when calling
    // `tensor_list_stack`) and also valid (not scalar), we rewrite this op to a
    // trivial Reshape op (that doesn't actually change the input's shape) and
    // also populate the shape info to the op result. The shape of the
    // tensorlist is inferred from `num_elements` and `element_shape`.
    auto ranked_type = element_shape->getType().dyn_cast<RankedTensorType>();
    DenseIntElementsAttr dense_elem_attr;
    if ((ranked_type && ranked_type.getRank() == 0) ||
        !matchPattern(element_shape, m_Constant(&dense_elem_attr))) {
      // If no constant is spotted, just forward the operand.
      rewriter.replaceOp(op, {input}, llvm::None);
      return matchSuccess();
    }

    RankedTensorType shape_type =
        RankedTensorType::get({-1}, rewriter.getIntegerType(32));
    auto new_shape = rewriter.create<TF::ShapeOp>(loc, shape_type, input);
    SmallVector<int64_t, 8> output_shape = {op.num_elements().getSExtValue()};
    for (auto dim : dense_elem_attr.getIntValues())
      output_shape.push_back(dim.getSExtValue());
    RankedTensorType result_type =
        RankedTensorType::get(output_shape, getElementTypeOrSelf(input));
    rewriter.replaceOpWithNewOp<TF::ReshapeOp>(op, result_type, input,
                                               new_shape);
    return matchSuccess();
  }
};

struct ConvertIdentity : public ConversionPattern {
  explicit ConvertIdentity(MLIRContext *context)
      : ConversionPattern(TF::IdentityOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *operation, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto op = llvm::cast<TF::IdentityOp>(operation);
    Value *input = operands[0];
    rewriter.replaceOpWithNewOp<TF::IdentityOp>(op, input->getType(), operands,
                                                op.getAttrs());
    return matchSuccess();
  }
};

// Changes the function type of `cond_func` and `body_func` for the given While
// op.
static LogicalResult UpdateFunctionTypes(TF::WhileOp op) {
  auto module = op.getParentOfType<ModuleOp>();
  auto *context = module.getContext();

  for (StringRef func_name : {op.cond(), op.body()}) {
    FuncOp func = module.lookupSymbol<FuncOp>(func_name);
    if (!func) continue;

    FunctionType func_type = func.getType();
    int num_inputs = func_type.getNumInputs();
    int num_results = func_type.getNumResults();

    // For each argument type in function's arguments, change it to uranked
    // tensor type if it's a variant type.
    SmallVector<Type, 8> updated_argument_types;
    updated_argument_types.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      Type arg_type = func_type.getInput(i);
      if (getElementTypeOrSelf(arg_type).isa<TF::VariantType>()) {
        arg_type = UnrankedTensorType::get(
            getElementTypeOrSelf(op.getOperand(i)->getType()));
      }
      updated_argument_types.push_back(arg_type);
    }

    // For each result type in function's results, change it to unranked tensor
    // type if it's a variant type.
    SmallVector<Type, 8> updated_result_types;
    updated_result_types.reserve(num_results);
    for (int i = 0; i < num_results; ++i) {
      Type result_type = func_type.getResult(i);
      if (getElementTypeOrSelf(result_type).isa<TF::VariantType>()) {
        // Here update the variant type with the unranked tensor type derived
        // from the corresponding input operand. This is correct because while
        // body's inputs and results have the same type.
        result_type = UnrankedTensorType::get(
            getElementTypeOrSelf(op.getOperand(i)->getType()));
      }
      updated_result_types.push_back(result_type);
    }

    // Change `func`'s argument type to `unranked_argument_types`. If it
    // return types contain a `DT_VARIANT`, change it to the unranked type
    // derived from the corresponding argument.
    func.setType(FunctionType::get(updated_argument_types, updated_result_types,
                                   context));

    // Change the argument type for the first block.
    Block &body_first_bb = func.front();
    for (int i = 0; i < body_first_bb.getNumArguments(); ++i) {
      body_first_bb.getArgument(i)->setType(updated_argument_types[i]);
    }
  }
  return success();
}

struct ConvertWhile : public ConversionPattern {
  explicit ConvertWhile(MLIRContext *context)
      : ConversionPattern(TF::WhileOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *operation, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto op = llvm::cast<TF::WhileOp>(operation);

    llvm::SmallVector<Type, 8> result_types;
    result_types.reserve(op.getNumOperands());
    for (int i = 0, e = operands.size(); i != e; ++i) {
      Type result_ty = op.getResult(i)->getType();

      // If we notice the result type is a DT_VARIANT, we change the
      // corresponding result type to unranked tensor type.
      if (getElementTypeOrSelf(result_ty).isa<TF::VariantType>()) {
        Type element_ty = getElementTypeOrSelf(operands[i]->getType());
        result_ty = UnrankedTensorType::get(element_ty);
      }
      result_types.push_back(result_ty);
    }

    // Clone original while op with new operands and updated result types.
    auto cloned = rewriter.create<TF::WhileOp>(op.getLoc(), result_types,
                                               operands, op.getAttrs());
    cloned.removeAttr("T");
    UpdateFunctionTypes(cloned);

    rewriter.replaceOp(op, cloned.getResults());
    return matchSuccess();
  }
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_lower_static_tensor_list.inc"

LogicalResult LowerStaticTensorListPass::RewriteFunction(
    FuncOp func, TensorListPatternRewriter *rewriter) {
  auto *context = &getContext();

  // TensorFlow operations that doesn't have operands and results of type
  // variant are legal. Here, we don't distinguish between variants encoding
  // TensorList or some other type as that information is not available here.
  // This constraint should be relaxed to support other variant types in TFLite.
  auto is_legal = [](Operation *op) {
    auto is_not_variant = [](Type ty) {
      return !ty.cast<ShapedType>().getElementType().isa<TF::VariantType>();
    };
    return llvm::all_of(op->getOperandTypes(), is_not_variant) &&
           llvm::all_of(op->getResultTypes(), is_not_variant);
  };

  ConversionTarget target(*context);
  target.addDynamicallyLegalDialect<TF::TensorFlowDialect>(
      llvm::Optional<ConversionTarget::DynamicLegalityCallbackFn>(is_legal));
  target.addIllegalOp<TF::EmptyTensorListOp, TF::TensorListFromTensorOp,
                      TF::TensorListGetItemOp, TF::TensorListLengthOp,
                      TF::TensorListPushBackOp, TF::TensorListReserveOp,
                      TF::TensorListSetItemOp, TF::TensorListStackOp,
                      TF::TensorListResizeOp>();
  // TODO(hinsu): Use TFLite constant op for constants.
  target.addLegalOp<ConstantOp>();
  target.addLegalOp<FuncOp>();
  target.addLegalOp<ReturnOp>();

  OwningRewritePatternList patterns;
  patterns
      .insert<ConvertEmptyTensorList, ConvertIdentity,
              ConvertTensorListFromTensor, ConvertTensorListGetItem,
              ConvertTensorListLength, ConvertTensorListPushBack,
              ConvertTensorListReserve, ConvertTensorListSetItem,
              ConvertTensorListStack, ConvertTensorListResize, ConvertWhile>(
          context);
  return applyFullConversion(func, target, patterns);
}

void LowerStaticTensorListPass::runOnModule() {
  // TODO(haoliang): currently we process the `main` function first, and the
  // remaining functions may be processed in arbitrary order. However, this will
  // have a potential issue when one function taking a `DT_VARIANT` is processed
  // before the function that produces the `DT_VARIANT`. We need to carefully
  // order the functions to be processed.
  std::vector<FuncOp> funcs_in_module;
  for (auto func : getModule().getOps<FuncOp>()) {
    // Always place the main function to be the first in the list.
    if (func.getName() == "main") {
      funcs_in_module.insert(funcs_in_module.begin(), func);
    } else {
      funcs_in_module.push_back(func);
    }
  }
  for (auto func : funcs_in_module) {
    TensorListPatternRewriter rewriter(func);
    if (failed(RewriteFunction(func, &rewriter))) {
      signalPassFailure();
      return;
    }
  }
}

}  // namespace

/// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
/// pass.
std::unique_ptr<OpPassBase<ModuleOp>> TFL::CreateLowerStaticTensorListPass() {
  return std::make_unique<LowerStaticTensorListPass>();
}

static PassRegistration<LowerStaticTensorListPass> pass(
    "tfl-lower-static-tensor-list",
    "Lower TensorList ops within TensorFlow Lite dialect");

}  // namespace mlir
