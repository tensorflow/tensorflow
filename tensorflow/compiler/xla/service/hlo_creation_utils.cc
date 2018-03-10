/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {
StatusOr<HloInstruction*> CreateBinaryHlo(HloOpcode opcode, HloInstruction* lhs,
                                          HloInstruction* rhs) {
  HloComputation* computation = lhs->parent();
  CHECK_EQ(computation, rhs->parent());
  TF_ASSIGN_OR_RETURN(Shape binary_op_shape,
                      ShapeInference::InferBinaryOpShape(opcode, lhs, rhs));
  return computation->AddInstruction(
      HloInstruction::CreateBinary(binary_op_shape, opcode, lhs, rhs));
}

StatusOr<HloInstruction*> CreatePadHlo(HloInstruction* operand,
                                       HloInstruction* padding_value,
                                       const PaddingConfig& padding_config) {
  HloComputation* computation = operand->parent();
  CHECK_EQ(computation, padding_value->parent());
  TF_ASSIGN_OR_RETURN(
      Shape pad_shape,
      ShapeInference::InferPadShape(operand->shape(), padding_value->shape(),
                                    padding_config));
  return computation->AddInstruction(HloInstruction::CreatePad(
      pad_shape, operand, padding_value, padding_config));
}

StatusOr<HloInstruction*> CreateSliceHlo(
    HloInstruction* operand, tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices,
    tensorflow::gtl::ArraySlice<int64> strides) {
  HloComputation* computation = operand->parent();
  TF_ASSIGN_OR_RETURN(Shape slice_shape, ShapeInference::InferSliceShape(
                                             operand->shape(), start_indices,
                                             limit_indices, strides));
  return computation->AddInstruction(HloInstruction::CreateSlice(
      slice_shape, operand, start_indices, limit_indices, strides));
}

StatusOr<HloInstruction*> CreateConvolveHlo(
    HloInstruction* lhs, HloInstruction* rhs, const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  HloComputation* computation = lhs->parent();
  CHECK_EQ(computation, rhs->parent());
  TF_ASSIGN_OR_RETURN(Shape convolve_shape, ShapeInference::InferConvolveShape(
                                                lhs->shape(), rhs->shape(),
                                                window, dimension_numbers));
  return computation->AddInstruction(HloInstruction::CreateConvolve(
      convolve_shape, lhs, rhs, window, dimension_numbers));
}

}  // namespace xla
