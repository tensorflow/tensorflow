/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Shape inference is used by the XLA service as the user builds up
// computation requests.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SHAPE_INFERENCE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SHAPE_INFERENCE_H_

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// For a given operation and input shapes, infers what the resulting shape is
// for the operation. With this functionality, the user does not need to specify
// the expected result type for computations that are built up via the API --
// the shape that results from an operation is inferred. Some methods have
// overloads for inferring shape at the HLO level.
//
// TODO(b/73352135): Shape inference does not issue very good error messages, in
// part because HloInstruction::ToString() is not available since shape
// inference runs before the HloInstruction object is created. We need a
// solution for this.
class ShapeInference {
 public:
  // Infers the shape produced by applying the given unary operation to the
  // given input shape.
  static StatusOr<Shape> InferUnaryOpShape(HloOpcode opcode,
                                           const Shape& shape);
  static StatusOr<Shape> InferUnaryOpShape(HloOpcode opcode,
                                           const HloInstruction* operand);

  // Infers the shape produced by applying the given binary operation to the
  // given input shapes.
  static StatusOr<Shape> InferBinaryOpShape(
      HloOpcode opcode, const Shape& lhs, const Shape& rhs,
      absl::Span<const int64> broadcast_dimensions);
  static StatusOr<Shape> InferBinaryOpShape(HloOpcode opcode,
                                            const HloInstruction* lhs,
                                            const HloInstruction* rhs);

  // Infers the shape produced by applying the given ternary operation to the
  // given input shapes.
  static StatusOr<Shape> InferTernaryOpShape(HloOpcode opcode, const Shape& lhs,
                                             const Shape& rhs,
                                             const Shape& ehs);
  static StatusOr<Shape> InferTernaryOpShape(HloOpcode opcode,
                                             const HloInstruction* lhs,
                                             const HloInstruction* rhs,
                                             const HloInstruction* ehs);

  // Infers the shape produced by applying the given variadic operation to the
  // given input operand shapes.
  static StatusOr<Shape> InferVariadicOpShape(
      HloOpcode opcode, absl::Span<const Shape* const> operand_shapes);
  static StatusOr<Shape> InferVariadicOpShape(
      HloOpcode opcode, absl::Span<const HloInstruction* const> operands);

  // Infers the shape produced by applying the given mapping computation shape
  // to the given operand shapes.
  static StatusOr<Shape> InferMapShape(
      absl::Span<const Shape* const> arg_shapes, const ProgramShape& to_apply,
      absl::Span<const int64> dimensions);

  // Infers the shape produced by InferBatchNormTraining with the given
  // operands.
  static StatusOr<Shape> InferBatchNormTrainingShape(const Shape& operand_shape,
                                                     const Shape& scale_shape,
                                                     const Shape& offset_shape,
                                                     int64 feature_index);

  // Infers the shape produced by InferBatchNormInference with the given
  // operands.
  static StatusOr<Shape> InferBatchNormInferenceShape(
      const Shape& operand_shape, const Shape& scale_shape,
      const Shape& offset_shape, const Shape& mean_shape,
      const Shape& variance_shape, int64 feature_index);

  // Infers the shape produced by InferBatchNormGrad with the given operands.
  static StatusOr<Shape> InferBatchNormGradShape(const Shape& operand_shape,
                                                 const Shape& scale_shape,
                                                 const Shape& mean_shape,
                                                 const Shape& var_shape,
                                                 const Shape& output_grad_shape,
                                                 int64 feature_index);

  // Infers the shape produced by applying the given convolutional filter (rhs)
  // to lhs in the way specified by the fields on window. An optional
  // preferred_element_type can be specified to upcast the element type.
  static StatusOr<Shape> InferConvolveShape(
      const Shape& lhs, const Shape& rhs, int64 feature_group_count,
      int64 batch_group_count, const Window& window,
      const ConvolutionDimensionNumbers& dimension_numbers,
      absl::optional<PrimitiveType> preferred_element_type);

  // Infers the shape produced by the given FFT type on the given operand.
  static StatusOr<Shape> InferFftShape(const Shape& in, FftType fft_type,
                                       absl::Span<const int64> fft_length);

  // Infers the shape produced by the given triangular solve operation.
  static StatusOr<Shape> InferTriangularSolveShape(
      const Shape& a, const Shape& b, const TriangularSolveOptions& options);

  // Infers the shape produced by the given triangular solve operation.
  static StatusOr<Shape> InferCholeskyShape(const Shape& a);

  // Infers the shape produced by an all-gather with the given operand shape,
  // concat dimension, and shard count.
  static StatusOr<Shape> InferAllGatherShape(
      absl::Span<const Shape* const> operand_shapes, int64 all_gather_dimension,
      int64 shard_count);

  // Infers the shape produced by a cross replica sum with the given operand
  // shapes.
  static StatusOr<Shape> InferAllReduceShape(
      absl::Span<const Shape* const> operand_shapes);

  // Infers final shape of an Alltoall operation that is created by the xla
  // builder.
  static StatusOr<Shape> InferAllToAllShape(const Shape& shape,
                                            int64 split_dimension,
                                            int64 concat_dimension,
                                            int64 split_count);

  // Infers the shape of an HLO all-to-all instruction.
  static StatusOr<Shape> InferAllToAllTupleShape(
      absl::Span<const Shape* const> operand_shapes);

  // Infers the shape of a collective permute operation.
  static StatusOr<Shape> InferCollectivePermuteShape(
      absl::Span<const Shape* const> operand_shapes);

  // Infers the shape of a collective permute start operation.
  static StatusOr<Shape> InferCollectivePermuteStartShape(
      absl::Span<const Shape* const> operand_shapes);

  // Infers the shape of a collective permute operation.
  static StatusOr<Shape> InferCollectivePermuteDoneShape(
      const Shape& operand_shape);

  // Infers the shape produced by applying the given reduction computation
  // shape to the given input operand shape.
  //
  // If pass_index is true, the reduce function is invoked with the element
  // index as the leading parameter, and the program shape should match
  // accordingly (or an error will result).
  static StatusOr<Shape> InferReduceShape(
      absl::Span<const Shape* const> arg_shapes,
      absl::Span<const int64> dimensions_to_reduce,
      const ProgramShape& to_apply);

  // Infers the shape produced by applying the given computation to the operand
  // shape with the given window and stride dimensions.
  static StatusOr<Shape> InferReduceWindowShape(
      const Shape& operand_shape, const Shape& init_value, const Window& window,
      const ProgramShape& to_apply_shape);
  static StatusOr<Shape> InferReduceWindowShape(const Shape& operand_shape,
                                                const Shape& init_value,
                                                const Window& window);
  static StatusOr<Shape> InferReduceWindowShape(
      absl::Span<const Shape* const> operands,
      absl::Span<const Shape* const> init_values, const Window& window,
      const ProgramShape& to_apply_shape);

  static StatusOr<Shape> InferReduceWindowShape(
      absl::Span<const Shape*> operands, absl::Span<const Shape*> init_values,
      const Window& window);

  // Infers the shape produced by scattering the given source shape to the
  // selected indices of each window on the operand shape.
  static StatusOr<Shape> InferSelectAndScatterShape(
      const Shape& operand_shape, const ProgramShape& select_shape,
      const Window& window, const Shape& source_shape,
      const Shape& init_value_shape, const ProgramShape& scatter_shape);

  // Infers the shape produced by a reverse operation that reverses the order
  // of the elements in the given dimensions.
  static StatusOr<Shape> InferReverseShape(const Shape& operand_shape,
                                           absl::Span<const int64> dimensions);

  // Infers the shape produced by a slice operation spanning from the starts to
  // the limits in the original shape's dimensions.
  //
  // e.g. slice f32[32x32] 0:16 0:16 -> f32[16x16]
  static StatusOr<Shape> InferSliceShape(const Shape& arg,
                                         absl::Span<const int64> starts,
                                         absl::Span<const int64> limits,
                                         absl::Span<const int64> strides);

  // Infers the shape produced by a dynamic slice operation of size specified
  // in 'slice_sizes', with dynamic start indices shape 'start_indices_shape'.
  static StatusOr<Shape> InferDynamicSliceShape(
      const Shape& operand_shape, absl::Span<const Shape> start_index_shapes,
      absl::Span<const int64> slice_sizes, bool allow_scalar_indices = true);

  // Infers the shape produced by a dynamic update slice operation based
  // on the shape of operand and update.
  static StatusOr<Shape> InferDynamicUpdateSliceShape(
      const Shape& operand_shape, const Shape& update_shape,
      absl::Span<const Shape> start_index_shapes,
      bool allow_scalar_indices = true);

  // Infers the shape produced by doing a compile-time-constant indexing into
  // the given input shape. This is essential for operations on tuples, because
  // it is impossible to infer the type that comes out of the tuple indexing if
  // it is not a compile time constant.
  static StatusOr<Shape> InferGetTupleElementShape(const Shape& arg,
                                                   int64 index);

  // Infers the shape produced from a while node. condition and body are the
  // shapes of computations for the condition and the body of a while node, and
  // init is the shape of data initially passed in to the body as an argument.
  // The shapes must match; condition: T -> PRED, body: T -> T, init: T
  static StatusOr<Shape> InferWhileShape(const ProgramShape& condition,
                                         const ProgramShape& body,
                                         const Shape& init);

  // Infers the shape produced by a predicated or indexed conditional operation.
  static StatusOr<Shape> InferConditionalShape(
      const Shape& branch_index,
      absl::Span<const ProgramShape> branch_computations,
      absl::Span<const Shape> branch_operands);

  // Infers the shape produced by a broadcast operation.
  static StatusOr<Shape> InferBroadcastShape(
      const Shape& operand, absl::Span<const int64> broadcast_sizes);

  // Checks whether the given parameters can form a broadcast. Returns the same
  // output_shape if it's legal.
  static StatusOr<Shape> InferBroadcastShape(
      const Shape& operand_shape, const Shape& output_shape,
      absl::Span<const int64> broadcast_dimensions);

  // Infers the shape produced by a reshape operation from the element type of
  // its operand and the new dimension sizes specified.
  static StatusOr<Shape> InferReshapeShape(const Shape& operand,
                                           absl::Span<const int64> dimensions,
                                           absl::Span<const int64> new_sizes,
                                           int64 inferred_dimension);

  // Infers the shape produced by a dynamic reshape operation from the element
  // type of its operand and the new dimension sizes specified. The result shape
  // will have dynamic dimensions as specific in `dim_is_dynamic` and bound
  // `new_size_bounds`.
  static StatusOr<Shape> InferDynamicReshapeShape(
      const Shape& operand, absl::Span<const Shape* const> dim_size_shapes,
      absl::Span<const int64> new_size_bounds,
      const std::vector<bool>& dims_are_dynamic);

  // Infers the shape produced by a transpose operation from the element type of
  // its operand and its dimensions field.
  static StatusOr<Shape> InferTransposeShape(
      const Shape& operand, absl::Span<const int64> dimensions);

  // Helper that infers the shape produced by performing a concatenate operation
  // with the given operand shapes.
  static StatusOr<Shape> InferConcatOpShape(
      absl::Span<const Shape* const> arg_shapes, int64 dimension);

  // Helper that validates the given operand shape can be converted to the
  // target output_shape via a convert instruction -- the requirement is that
  // the shape is identical except for the element type.
  static StatusOr<Shape> InferConvertShape(const Shape& operand_shape,
                                           PrimitiveType new_element_type);

  // Helper that validates the given operand shape can be bitcast converted to
  // the target output_shape via a bitcast convert instruction -- the
  // requirement is that the shape is identical except for the element type and
  // the element types have identical bit-widths.
  static StatusOr<Shape> InferBitcastConvertShape(
      const Shape& operand_shape, PrimitiveType new_element_type);

  // Helper that validates the input data type for a reduce-precision operation,
  // and returns the result shape.
  static StatusOr<Shape> InferReducePrecisionShape(const Shape& operand_shape,
                                                   const int exponent_bits,
                                                   const int mantissa_bits);

  // Helper that infers the shape produced by a pad operation based on the
  // padding configuration.
  static StatusOr<Shape> InferPadShape(const Shape& operand_shape,
                                       const Shape& padding_value_shape,
                                       const PaddingConfig& padding_config);

  // Helper that validates the given arg_shapes are compatible with the shape of
  // the to_apply parameters, and returns the to_apply result shape.
  static StatusOr<Shape> InferCallShape(
      absl::Span<const Shape* const> arg_shapes, const ProgramShape& to_apply);

  // Helper that infers the shape produced by performing a dot operation with
  // the given LHS and RHS shapes. An optional preferred_element_type can be
  // specified to upcast the element type.
  static StatusOr<Shape> InferDotOpShape(
      const Shape& lhs, const Shape& rhs,
      const DotDimensionNumbers& dimension_numbers,
      absl::optional<PrimitiveType> preferred_element_type);

  // Helper that infers the shape of the tensor produced by a gather operation
  // with the given input shape, gather indices shape and gather dimension
  // numbers.
  static StatusOr<Shape> InferGatherShape(
      const Shape& input_shape, const Shape& start_indices_shape,
      const GatherDimensionNumbers& gather_dim_numbers,
      absl::Span<const int64> slice_sizes);

  // Helper that validates the given input shape, scatter indices shape, updates
  // shape, and scatter dimension numbers that constitute a scatter operation,
  // and returns the result shape of the scatter operation.
  static StatusOr<Shape> InferScatterShape(
      const Shape& operand_shape, const Shape& scatter_indices_shape,
      const Shape& updates_shape, const ProgramShape& to_apply_shape,
      const ScatterDimensionNumbers& scatter_dim_numbers);

  // Helper that validates the given input shape to GetDimensionSize.
  static StatusOr<Shape> InferGetDimensionSizeShape(const Shape& shape,
                                                    int64 dimension);

  // Helper that validates the given input shape to SetDimensionSize.
  static StatusOr<Shape> InferSetDimensionSizeShape(const Shape& operand_shape,
                                                    const Shape& val_shape,
                                                    int64 dimension);

  // Helper function for creating a Window proto from user-supplied data.
  // Returns error if the user-supplied data was invalid.
  static StatusOr<Window> InferWindowFromDimensions(
      absl::Span<const int64> window_dimensions,
      absl::Span<const int64> window_strides,
      absl::Span<const std::pair<int64, int64>> padding,
      absl::Span<const int64> lhs_dilation,
      absl::Span<const int64> rhs_dilation);

 private:
  // Helper that infers the shape produced by performing an element-wise binary
  // operation with the given LHS and RHS shapes.
  // Note: By "element-wise" we mean operations that look at a single element in
  // the LHS and a single element in the RHS to produce a single output element,
  // even in the presence of broadcasting of one of the operands over the other.
  static StatusOr<Shape> InferElementwiseBinaryOpShape(
      HloOpcode operation, const Shape& lhs, const Shape& rhs,
      absl::Span<const int64> broadcast_dimensions);

  // Helper for inferring the shape of Clamp ops.
  static StatusOr<Shape> InferClampShape(const Shape& min, const Shape& operand,
                                         const Shape& max);

  // Helper for inferring the shape of Select ops.
  static StatusOr<Shape> InferSelectShape(const Shape& pred,
                                          const Shape& on_true,
                                          const Shape& on_false);
  // Helper for inferring the shape of TupleSelect ops.
  static StatusOr<Shape> InferTupleSelectShape(const Shape& pred,
                                               const Shape& on_true,
                                               const Shape& on_false);

  // Helper for inferring shapes of binary operations which use degenerate
  // dimension broadcasting (a dimension of size 1 in one operand is broadcast
  // up to match the size of the dimension in the other operand).
  static StatusOr<Shape> InferDegenerateDimensionBroadcastShape(
      HloOpcode operation, const Shape& lhs, const Shape& rhs);

  // Helper for inferring shapes of binary operations using "InDim"
  // broadcasting. This is the broadcasting used in the *InDim binary operations
  // (for example ComputationBuilder::AddInDim). smaller_shape must be a
  // lower-rank shape than larger_shape. Returns the shape that the
  // smaller_shape is broadcast to.
  static StatusOr<Shape> InferInDimBroadcastShape(
      const Shape& smaller_shape, const Shape& larger_shape,
      absl::Span<const int64> broadcast_dimensions);

  TF_DISALLOW_COPY_AND_ASSIGN(ShapeInference);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SHAPE_INFERENCE_H_
