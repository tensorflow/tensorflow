/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_UTIL_H_

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

struct PermutationAndShape {
  DenseIntElementsAttr permutation;
  ShapedType shape;
};

// Check that `arr` is an R1 iota with integer element type starting from `0`
// with `size` number of values.
bool IsIotaAttr(ArrayRef<int64_t> arr, int64_t size);

// Returns a DenseIntElementsAttr for a permutation and the shape after
// applying the permutation to a given shape through a transpose.
PermutationAndShape GetPermutationAndTransposedShape(
    llvm::ArrayRef<int64_t> permutation_array, ShapedType input_type,
    ConversionPatternRewriter& rewriter);

// Create a single const integer.
Value BuildIntConstOp(ImplicitLocOpBuilder& builder,
                      ConversionPatternRewriter& rewriter, int64_t const_value,
                      Type type);

// Create a const integer vector tensor (1-dim).
Value BuildIntArrayConstOp(ImplicitLocOpBuilder& builder,
                           ConversionPatternRewriter& rewriter,
                           ArrayRef<int64_t> const_value, Type type);

// Returns the inverse permutation array for a permutation array.
llvm::SmallVector<int64_t> GetInversePermutationArray(
    llvm::ArrayRef<int64_t> permutation_array);

// Returns the DenseIntElementsAttr for an inverse permutation given a
// permutation_array.
DenseIntElementsAttr GetInversePermutation(
    llvm::ArrayRef<int64_t> permutation_array,
    ConversionPatternRewriter& rewriter);

// Returns a DenseIntElementsAttr for an inverse permutation and the shape after
// applying the inverse permutation to a given shape through a transpose.
PermutationAndShape GetInversePermutationAndShape(
    llvm::ArrayRef<int64_t> permutation_array, ShapedType input_type,
    ConversionPatternRewriter& rewriter);

// Returns true if the op needs reformat.
bool NeedsReformatTypeAndPermutation(int batch_dim, int feature_dim,
                                     int spatial_dim_start,
                                     int default_batch_dim,
                                     int default_feature_dim,
                                     int default_spatial_dim_start);

// Gets reformat type and permutation attribute. Call this function only if
// NeedsReformatTypeAndPermutation returns true. If
// NeedsReformatTypeAndPermutation returns false, this function returns the pair
// of input type and no-op permutation.

std::pair<RankedTensorType, DenseIntElementsAttr> GetReformatTypeAndPermutation(
    int batch_dim, int feature_dim, int spatial_dim_start,
    int default_batch_dim, int default_feature_dim,
    int default_spatial_dim_start, int num_spatial_dims, RankedTensorType type,
    ConversionPatternRewriter& rewriter);

// Insert transpose so the input value is converted to the format specified by
// the default dims
Value InsertTranspose(Value value, int batch_dim, int feature_dim,
                      ArrayRef<int64_t> spatial_dimensions,
                      int default_batch_dim, int default_feature_dim,
                      int default_spatial_dim_start, int num_spatial_dims,
                      ConversionPatternRewriter& rewriter);

// If index_vector_dim == indices.rank() then insert the implicit extra
// dimension into indices to normalize everything to index_vector_dim ==
// indices.rank() - 1.
LogicalResult NormalizeIndexVector(Operation* parent_op, Value& indices,
                                   ShapedType& indices_type,
                                   int64_t index_vector_dim,
                                   ConversionPatternRewriter& rewriter);

// Checks if the specified region is a binary reduction function that takes 2
// inputs, passes it to an instance of the specified reduction op and then
// returns the result.
template <typename ReductionOp>
LogicalResult MatchBinaryReduceFunction(mlir::Region& function) {
  Block& body = function.front();
  if (body.getNumArguments() != 2) return failure();

  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op) return failure();
  if (return_op.getNumOperands() != 1) return failure();

  ReductionOp reduce_op = dyn_cast_or_null<ReductionOp>(
      return_op.getOperands().front().getDefiningOp());
  if (!reduce_op) return failure();
  if (reduce_op.getLhs() != body.getArgument(0) ||
      reduce_op.getRhs() != body.getArgument(1))
    return failure();

  return success();
}

// Check if the specified region is a binary reduction function that takes 2
// inputs and returns the second input. Functions like this are used by update
// scatter like ops.
template <>
LogicalResult MatchBinaryReduceFunction<void>(mlir::Region& function);

// Concentrates the data needed to substitute StableHLO operations with TFLite
// ones.
struct ConversionState {
  Operation* hlo_op;
  ConversionPatternRewriter& rewriter;
  Operation* last_tf_op;

  // Returns the main operand of a NEW op to add to the conversion chain.
  //
  // This is generally the result of the last op that was added to the chain.
  Value GetOperand() const;

  // Returns the type of the operand of a NEW op to add to the conversion chain.
  //
  // This is generally the type of the result of the last op that was added to
  // the chain.
  TensorType GetOperandTensorType() const;

  llvm::ArrayRef<int64_t> GetOperandShape() const;

  // Computes a new shape from the current operand shape.
  //
  // - The args are containers that are indexable using operator[].
  // - The callback must be callable have a signature that is:
  //      `int64_t (int idx, shape, decltype(args)...)`
  //
  // The callback is called for each element of the operand shape with the
  // index of the current loop iteration, the shape and args.
  template <class F, class... Containers>
  llvm::SmallVector<int64_t, 6> ComputeResultShape(F&& callback,
                                                   Containers&&... args) const {
    llvm::ArrayRef<int64_t> shape = GetOperandShape();
    llvm::SmallVector<int64_t, 6> res;
    for (int i = 0; i < shape.size(); ++i) {
      if (shape[i] < 0) {
        res.push_back(shape[i]);
      } else {
        res.push_back(callback(i, shape, args...));
      }
    }
    return res;
  }

  template <class F, class... Containers>
  TensorType ComputeResultTensorType(F&& callback, Containers&&... args) const {
    const llvm::SmallVector<int64_t, 6> shape = ComputeResultShape(
        static_cast<F&&>(callback), static_cast<Containers&&>(args)...);
    return GetOperandTensorType().cloneWith(
        shape, GetOperandTensorType().getElementType());
  }
};

// Gets the Type associated to type T from the builder.
template <class T>
Type GetElementType(OpBuilder& builder);

#define GET_ELEMENT_TYPE_SPECIALISATION(TYPE, NAME)       \
  template <>                                             \
  inline Type GetElementType<TYPE>(OpBuilder & builder) { \
    return builder.get##NAME##Type();                     \
  }

GET_ELEMENT_TYPE_SPECIALISATION(int32_t, I32);
GET_ELEMENT_TYPE_SPECIALISATION(int64_t, I64);

// Create a DenseElementsAttr from given shape and data.
template <class Data, class Shape = llvm::SmallVector<int64_t, 6>>
DenseElementsAttr CreateDenseElementsAttr(OpBuilder& builder, const Data& data,
                                          const Shape& shape = Shape()) {
  llvm::SmallVector<int64_t, 6> attr_shape(shape.begin(), shape.end());
  if (attr_shape.empty()) {
    attr_shape.push_back(static_cast<int64_t>(data.size()));
  }
  const Type attr_type = GetElementType<typename Data::value_type>(builder);
  return DenseElementsAttr::get(RankedTensorType::get(attr_shape, attr_type),
                                ArrayRef<typename Data::value_type>(data));
}

// Adds a constant tensor to the conversion chain.
template <class Data, class Shape = llvm::SmallVector<int64_t, 6>>
auto AddConstantTensor(ConversionState& state, const Data& data,
                       const Shape& shape = Shape()) {
  const DenseElementsAttr attr =
      CreateDenseElementsAttr(state.rewriter, data, shape);
  return state.rewriter.create<arith::ConstantOp>(state.hlo_op->getLoc(), attr);
}

// Builds a callable object that checks that its argument is not the given
// `value`.
template <class T>
auto IsNot(T value) {
  return [value](auto v) { return v != value; };
}

// Adds a TFLite Dilate operation to the conversion chain.
//
// If the given parameters would end with the identity operation, this does not
// add anything to the chain.
//
// Depending on the definition of the op we are trying to legalize, a dilation
// can be either seen as interior padding or as a scaling factor where:
//
//     scaling_factor = interior_padding + 1
//
// The is_padding parameter is used to take this difference into account.
void AddDilateOpIfRequired(ConversionState& state,
                           const DenseElementsAttr& dilation,
                           Value padding_value, bool is_padding);

// Adds a TFLite PadV2 operation to the conversion chain.
//
// If the given parameters would end with the identity operation, this does not
// add anything to the chain.
void AddPadOpIfRequired(ConversionState& state,
                        const DenseElementsAttr& edge_padding_low,
                        const DenseElementsAttr& edge_padding_high,
                        Value padding_value);

// Adds a TFLite StridedSlice operation to the conversion chain.
//
// This overload is used to legalize a crop operation in TFLite. As such, the
// begin and end specifications of the strided slice are computed from the
// negative values in the padding parameters.
//
// If the given parameters would end with the identity operation, this does not
// add anything to the chain.
void AddStridedSliceOpIfRequired(ConversionState& state,
                                 const DenseElementsAttr& edge_padding_low,
                                 const DenseElementsAttr& edge_padding_high,
                                 const DenseElementsAttr& strides);

// Util that casts 'val' to Int32 by adding a tfl cast Op.
Value CreateCastToInt32(Value val, Location loc, PatternRewriter& rewriter);
}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_UTIL_H_
