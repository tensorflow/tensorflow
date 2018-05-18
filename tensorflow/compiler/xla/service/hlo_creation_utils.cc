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
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
using tensorflow::gtl::ArraySlice;
using tensorflow::strings::StrCat;

StatusOr<HloInstruction*> MakeBinaryHlo(HloOpcode opcode, HloInstruction* lhs,
                                        HloInstruction* rhs) {
  HloComputation* computation = lhs->parent();
  CHECK_EQ(computation, rhs->parent());
  TF_ASSIGN_OR_RETURN(Shape binary_op_shape,
                      ShapeInference::InferBinaryOpShape(opcode, lhs, rhs));
  return computation->AddInstruction(
      HloInstruction::CreateBinary(binary_op_shape, opcode, lhs, rhs));
}

StatusOr<HloInstruction*> MakePadHlo(HloInstruction* operand,
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

StatusOr<HloInstruction*> MakeSliceHlo(HloInstruction* operand,
                                       ArraySlice<int64> start_indices,
                                       ArraySlice<int64> limit_indices,
                                       ArraySlice<int64> strides) {
  HloComputation* computation = operand->parent();
  TF_ASSIGN_OR_RETURN(Shape slice_shape, ShapeInference::InferSliceShape(
                                             operand->shape(), start_indices,
                                             limit_indices, strides));
  return computation->AddInstruction(HloInstruction::CreateSlice(
      slice_shape, operand, start_indices, limit_indices, strides));
}

StatusOr<HloInstruction*> MakeConvolveHlo(
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

StatusOr<HloInstruction*> MakeTransposeHlo(HloInstruction* operand,
                                           ArraySlice<int64> dimensions) {
  HloComputation* computation = operand->parent();
  TF_ASSIGN_OR_RETURN(
      Shape transpose_shape,
      ShapeInference::InferTransposeShape(operand->shape(), dimensions));
  return computation->AddInstruction(
      HloInstruction::CreateTranspose(transpose_shape, operand, dimensions));
}

StatusOr<HloInstruction*> MakeReshapeHlo(const Shape& result_shape,
                                         HloInstruction* operand) {
  HloComputation* computation = operand->parent();
  return computation->AddInstruction(
      HloInstruction::CreateReshape(result_shape, operand));
}

StatusOr<HloInstruction*> MakeReshapeHlo(
    ArraySlice<int64> result_shape_dim_bounds, HloInstruction* operand) {
  Shape new_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                         result_shape_dim_bounds);
  return MakeReshapeHlo(new_shape, operand);
}

StatusOr<HloInstruction*> MakeDynamicSliceHlo(HloInstruction* operand,
                                              HloInstruction* start_indices,
                                              ArraySlice<int64> slice_sizes) {
  HloComputation* computation = operand->parent();
  CHECK_EQ(computation, start_indices->parent());
  TF_ASSIGN_OR_RETURN(
      Shape dynamic_slice_shape,
      ShapeInference::InferDynamicSliceShape(
          operand->shape(), start_indices->shape(), slice_sizes));
  return computation->AddInstruction(HloInstruction::CreateDynamicSlice(
      dynamic_slice_shape, operand, start_indices, slice_sizes));
}

StatusOr<HloInstruction*> MakeDynamicUpdateSliceHlo(
    HloInstruction* operand, HloInstruction* update,
    HloInstruction* start_indices) {
  HloComputation* computation = operand->parent();
  CHECK_EQ(computation, update->parent());
  CHECK_EQ(computation, start_indices->parent());
  TF_ASSIGN_OR_RETURN(
      Shape dynamic_update_slice_shape,
      ShapeInference::InferDynamicUpdateSliceShape(
          operand->shape(), update->shape(), start_indices->shape()));
  return computation->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      dynamic_update_slice_shape, operand, update, start_indices));
}

StatusOr<HloInstruction*> MakeBroadcastHlo(
    HloInstruction* operand, ArraySlice<int64> broadcast_dimensions,
    ArraySlice<int64> result_shape_bounds) {
  HloComputation* computation = operand->parent();
  Shape broadcast_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                               result_shape_bounds);

  return computation->AddInstruction(HloInstruction::CreateBroadcast(
      broadcast_shape, operand, broadcast_dimensions));
}

StatusOr<HloInstruction*> MakeGetTupleElementHlo(HloInstruction* operand,
                                                 int64 index) {
  HloComputation* computation = operand->parent();

  TF_ASSIGN_OR_RETURN(
      Shape gte_shape,
      ShapeInference::InferGetTupleElementShape(operand->shape(), index));
  return computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(gte_shape, operand, index));
}

StatusOr<HloInstruction*> MakeConcatHlo(ArraySlice<HloInstruction*> operands,
                                        int64 dimension) {
  CHECK_GT(operands.size(), 0);

  HloComputation* computation = operands[0]->parent();
  CHECK(c_all_of(operands, [&](HloInstruction* instr) {
    return instr->parent() == computation;
  }));

  std::vector<const Shape*> operand_shapes;
  c_transform(operands, std::back_inserter(operand_shapes),
              [](HloInstruction* instr) { return &instr->shape(); });

  TF_ASSIGN_OR_RETURN(Shape concat_shape, ShapeInference::InferConcatOpShape(
                                              operand_shapes, dimension));
  return computation->AddInstruction(
      HloInstruction::CreateConcatenate(concat_shape, operands, dimension));
}

StatusOr<HloInstruction*> MakeDotHlo(HloInstruction* lhs, HloInstruction* rhs,
                                     const DotDimensionNumbers& dim_numbers) {
  HloComputation* computation = lhs->parent();
  CHECK_EQ(computation, rhs->parent());
  TF_ASSIGN_OR_RETURN(
      Shape dot_shape,
      ShapeInference::InferDotOpShape(lhs->shape(), rhs->shape(), dim_numbers));
  return computation->AddInstruction(
      HloInstruction::CreateDot(dot_shape, lhs, rhs, dim_numbers));
}

StatusOr<HloInstruction*> CollapseFirstNDims(HloInstruction* operand, int64 n) {
  CHECK_GT(n, 0);

  const Shape& operand_shape = operand->shape();
  CHECK_GE(operand_shape.dimensions_size(), n);
  int64 new_shape_leading_bound = 1;
  for (int64 i = 0; i < n; i++) {
    new_shape_leading_bound *= operand_shape.dimensions(i);
  }

  std::vector<int64> new_shape_dims;
  new_shape_dims.reserve(operand_shape.dimensions_size() - n + 1);
  new_shape_dims.push_back(new_shape_leading_bound);

  std::copy(operand_shape.dimensions().begin() + n,
            operand_shape.dimensions().end(),
            std::back_inserter(new_shape_dims));

  Shape output_shape =
      ShapeUtil::MakeShape(operand_shape.element_type(), new_shape_dims);

  return MakeReshapeHlo(output_shape, operand);
}

StatusOr<HloInstruction*> PrependDegenerateDims(HloInstruction* operand,
                                                int64 n) {
  CHECK_GT(n, 0);
  std::vector<int64> new_shape_dims;
  const Shape& operand_shape = operand->shape();
  new_shape_dims.reserve(n + operand_shape.dimensions_size());
  new_shape_dims.insert(new_shape_dims.begin(), n, 1);
  c_copy(operand_shape.dimensions(), std::back_inserter(new_shape_dims));
  return MakeReshapeHlo(new_shape_dims, operand);
}

StatusOr<HloInstruction*> ExpandFirstDimIntoNDims(
    HloInstruction* operand, ArraySlice<int64> expanded_dims) {
  CHECK_GT(operand->shape().dimensions_size(), 0);
  CHECK_EQ(operand->shape().dimensions(0), Product(expanded_dims));

  std::vector<int64> expanded_shape_dim_bounds;
  expanded_shape_dim_bounds.reserve(expanded_dims.size() +
                                    operand->shape().dimensions_size() - 1);
  c_copy(expanded_dims, std::back_inserter(expanded_shape_dim_bounds));
  std::copy(operand->shape().dimensions().begin() + 1,
            operand->shape().dimensions().end(),
            std::back_inserter(expanded_shape_dim_bounds));
  Shape new_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                         expanded_shape_dim_bounds);
  return MakeReshapeHlo(new_shape, operand);
}

StatusOr<HloInstruction*> ElideDegenerateDims(HloInstruction* operand,
                                              ArraySlice<int64> dims_to_elide) {
  CHECK(c_is_sorted(dims_to_elide));

  const Shape& input_shape = operand->shape();
  // First accumulate in reverse
  std::vector<int64> new_shape_dim_bounds;
  new_shape_dim_bounds.reserve(input_shape.dimensions_size() -
                               dims_to_elide.size());
  int64 dims_to_elide_idx = dims_to_elide.size() - 1;
  for (int64 i = input_shape.dimensions_size() - 1; i >= 0; i--) {
    if (dims_to_elide_idx >= 0 && i == dims_to_elide[dims_to_elide_idx]) {
      CHECK_EQ(input_shape.dimensions(i), 1);
      dims_to_elide_idx--;
    } else {
      new_shape_dim_bounds.push_back(input_shape.dimensions(i));
    }
  }

  c_reverse(new_shape_dim_bounds);
  Shape output_shape =
      ShapeUtil::MakeShape(input_shape.element_type(), new_shape_dim_bounds);
  return MakeReshapeHlo(output_shape, operand);
}

StatusOr<HloInstruction*> PadVectorWithZeros(HloInstruction* operand,
                                             int64 zeros_to_prepend,
                                             int64 zeros_to_append) {
  HloComputation* computation = operand->parent();
  CHECK_EQ(operand->shape().dimensions_size(), 1);
  PaddingConfig padding_config;
  PaddingConfig::PaddingConfigDimension padding_config_dim;
  padding_config_dim.set_edge_padding_low(zeros_to_prepend);
  padding_config_dim.set_edge_padding_high(zeros_to_append);
  *padding_config.add_dimensions() = padding_config_dim;

  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          MakeUnique<Literal>(Literal::Zero(operand->shape().element_type()))));
  return MakePadHlo(operand, zero, padding_config);
}

StatusOr<HloInstruction*> BroadcastZeros(
    HloComputation* computation, PrimitiveType element_type,
    ArraySlice<int64> broadcast_dimensions) {
  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          MakeUnique<Literal>(Literal::Zero(element_type))));
  return MakeBroadcastHlo(zero, /*broadcast_dimensions=*/{},
                          /*result_shape_bounds=*/broadcast_dimensions);
}

StatusOr<std::unique_ptr<HloComputation>> CreateComputationWithSignature(
    ArraySlice<const Shape*> domain, const Shape& range,
    tensorflow::StringPiece name) {
  HloComputation::Builder b{std::string(name)};
  int64 param_idx = 0;
  for (const Shape* param_shape : domain) {
    b.AddInstruction(HloInstruction::CreateParameter(
        param_idx, *param_shape, StrCat("param.", param_idx)));
    param_idx++;
  }

  // We can't change the root type of a computation once it is created so create
  // a dummy root instruction to give the computation the right root shape.  In
  // the future we may want to use a (recursive) broadcast here to avoid
  // creating large constants.
  b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(range)));

  return b.Build();
}

}  // namespace xla
