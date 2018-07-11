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

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
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
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
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
      HloOpcode opcode,
      tensorflow::gtl::ArraySlice<const Shape*> operand_shapes);
  static StatusOr<Shape> InferVariadicOpShape(
      HloOpcode opcode,
      tensorflow::gtl::ArraySlice<const HloInstruction*> operands);

  // Infers the shape produced by applying the given mapping computation shape
  // to the given operand shapes.
  static StatusOr<Shape> InferMapShape(
      tensorflow::gtl::ArraySlice<const Shape*> arg_shapes,
      const ProgramShape& to_apply,
      tensorflow::gtl::ArraySlice<int64> dimensions);

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

  // Infers the shape produced by applying the given convolutional
  // filter (rhs) to lhs in the way specified by the fields on window.
  static StatusOr<Shape> InferConvolveShape(
      const Shape& lhs, const Shape& rhs, const Window& window,
      const ConvolutionDimensionNumbers& dimension_numbers);

  // Infers the shape produced by the given FFT type on the given operand.
  static StatusOr<Shape> InferFftShape(
      const Shape& in, FftType fft_type,
      tensorflow::gtl::ArraySlice<int64> fft_length);

  // Infers the shape produced a cross replica sum with the given operand
  // shapes.
  static StatusOr<Shape> InferCrossReplicaSumShape(
      tensorflow::gtl::ArraySlice<const Shape*> operand_shapes);

  // Infers the shape produced by applying the given reduction computation
  // shape to the given input operand shape.
  //
  // If pass_index is true, the reduce function is invoked with the element
  // index as the leading parameter, and the program shape should match
  // accordingly (or an error will result).
  static StatusOr<Shape> InferReduceShape(
      const Shape& arg, const Shape& init_value,
      tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
      const ProgramShape& to_apply);

  // Infers the shape produced by applying the given computation to the operand
  // shape with the given window and stride dimensions.
  static StatusOr<Shape> InferReduceWindowShape(
      const Shape& operand_shape, const Shape& init_value, const Window& window,
      const ProgramShape& to_apply_shape);

  // Infers the shape produced by scattering the given source shape to the
  // selected indices of each window on the operand shape.
  static StatusOr<Shape> InferSelectAndScatterShape(
      const Shape& operand_shape, const ProgramShape& select_shape,
      const Window& window, const Shape& source_shape,
      const Shape& init_value_shape, const ProgramShape& scatter_shape);

  // Infers the shape produced by a reverse operation that reverses the order
  // of the elements in the given dimensions.
  static StatusOr<Shape> InferReverseShape(
      const Shape& operand_shape,
      tensorflow::gtl::ArraySlice<int64> dimensions);

  // Infers the shape produced by a slice operation spanning from the starts to
  // the limits in the original shape's dimensions.
  //
  // e.g. slice f32[32x32] 0:16 0:16 -> f32[16x16]
  static StatusOr<Shape> InferSliceShape(
      const Shape& arg, tensorflow::gtl::ArraySlice<int64> starts,
      tensorflow::gtl::ArraySlice<int64> limits,
      tensorflow::gtl::ArraySlice<int64> strides);

  // Infers the shape produced by a dynamic slice operation of size specified
  // in 'slice_sizes', with dynamic start indices shape 'start_indices_shape'.
  static StatusOr<Shape> InferDynamicSliceShape(
      const Shape& operand_shape, const Shape& start_indices_shape,
      tensorflow::gtl::ArraySlice<int64> slice_sizes);

  // Infers the shape produced by a dynamic update slice operation based
  // on the shape of operand and update.
  static StatusOr<Shape> InferDynamicUpdateSliceShape(
      const Shape& operand_shape, const Shape& update_shape,
      const Shape& start_indices_shape);

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

  // Infers the shape produced by a conditional operation.
  static StatusOr<Shape> InferConditionalShape(
      const Shape& predicate, const Shape& true_operand,
      const Shape& false_operand, const ProgramShape& true_computation,
      const ProgramShape& false_computation);

  // Infers the shape produced by a broadcast operation.
  static StatusOr<Shape> InferBroadcastShape(
      const Shape& operand, tensorflow::gtl::ArraySlice<int64> broadcast_sizes);

  // Infers the shape produced by a reshape operation from the element type of
  // its operand and the new dimension sizes specified.
  static StatusOr<Shape> InferReshapeShape(
      const Shape& operand, tensorflow::gtl::ArraySlice<int64> dimensions,
      tensorflow::gtl::ArraySlice<int64> new_sizes);

  // Infers the shape produced by a transpose operation from the element type of
  // its operand and its dimensions field.
  static StatusOr<Shape> InferTransposeShape(
      const Shape& operand, tensorflow::gtl::ArraySlice<int64> dimensions);

  // Helper that infers the shape produced by performing a concatenate operation
  // with the given operand shapes.
  static StatusOr<Shape> InferConcatOpShape(
      tensorflow::gtl::ArraySlice<const Shape*> arg_shapes, int64 dimension);

  // Infers the shape produced by a kAfterAll. Trivially this shape is always a
  // TOKEN shape. However, ShapeInference serves two purposes: inferring shapes
  // and checking operand shapes. This method verifies that the operand shapes
  // are all TOKENs.
  static StatusOr<Shape> InferAfterAllShape(
      tensorflow::gtl::ArraySlice<const Shape*> arg_shapes);

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
      tensorflow::gtl::ArraySlice<const Shape*> arg_shapes,
      const ProgramShape& to_apply);

  // Helper that infers the shape produced by performing a dot operation with
  // the given LHS and RHS shapes.
  static StatusOr<Shape> InferDotOpShape(
      const Shape& lhs, const Shape& rhs,
      const DotDimensionNumbers& dimension_numbers);

  // Helper that infers the shape of the tensor produced by a gather operation
  // with the given input shape, gather indices shape and gather dimension
  // numbers.
  static StatusOr<Shape> InferGatherShape(
      const Shape& input_shape, const Shape& gather_indices_shape,
      const GatherDimensionNumbers& gather_dim_numbers,
      tensorflow::gtl::ArraySlice<int64> window_bounds);

 private:
  // Helper that infers the shape produced by performing an element-wise binary
  // operation with the given LHS and RHS shapes.
  // Note: By "element-wise" we mean operations that look at a single element in
  // the LHS and a single element in the RHS to produce a single output element,
  // even in the presence of broadcasting of one of the operands over the other.
  static StatusOr<Shape> InferElementwiseBinaryOpShape(
      HloOpcode operation, const Shape& lhs, const Shape& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);

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
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);

  TF_DISALLOW_COPY_AND_ASSIGN(ShapeInference);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SHAPE_INFERENCE_H_
