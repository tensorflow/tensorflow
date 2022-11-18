/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/conv_layout_normalization.h"

#include <optional>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"

namespace xla {
namespace gpu {

StatusOr<std::optional<HloInstruction*>>
NormalizeLayoutForCustomCallConvolution(HloCustomCallInstruction* hlo) {
  if (!IsCustomCallToDnnConvolution(*hlo)) {
    return {std::nullopt};
  }

  HloInstruction* lhs = hlo->mutable_operand(0);
  HloInstruction* rhs = hlo->mutable_operand(1);
  const ConvolutionDimensionNumbers& dim_numbers =
      hlo->convolution_dimension_numbers();

  auto transpose_dim = [&](int64_t dim, const Shape& unnormalized_shape) {
    return unnormalized_shape.rank() -
           FindIndex(unnormalized_shape.layout().minor_to_major(), dim) - 1;
  };

  auto transpose_dims = [&](tsl::protobuf::RepeatedField<int64_t>& dims,
                            const Shape& unnormalized_shape) {
    for (auto& dim : dims) {
      dim = transpose_dim(dim, unnormalized_shape);
    }
  };

  const Shape& conv_output_shape =
      hlo->shape().IsTuple() ? hlo->shape().tuple_shapes(0) : hlo->shape();

  Shape input_shape, filter_shape, output_shape;
  TF_ASSIGN_OR_RETURN(
      gpu::CudnnConvKind conv_kind,
      gpu::GetCudnnConvKind(Cast<HloCustomCallInstruction>(hlo)));
  switch (conv_kind) {
    case gpu::CudnnConvKind::kForward:
    case gpu::CudnnConvKind::kForwardActivation: {
      input_shape = lhs->shape();
      filter_shape = rhs->shape();
      output_shape = conv_output_shape;
      break;
    }
    case gpu::CudnnConvKind::kBackwardInput: {
      filter_shape = rhs->shape();
      output_shape = lhs->shape();
      input_shape = conv_output_shape;
      break;
    }
    case gpu::CudnnConvKind::kBackwardFilter: {
      input_shape = lhs->shape();
      output_shape = rhs->shape();
      filter_shape = conv_output_shape;
      break;
    }
  }

  ConvolutionDimensionNumbers new_dim_numbers = dim_numbers;
  new_dim_numbers.set_input_batch_dimension(
      transpose_dim(dim_numbers.input_batch_dimension(), input_shape));
  new_dim_numbers.set_input_feature_dimension(
      transpose_dim(dim_numbers.input_feature_dimension(), input_shape));
  transpose_dims(*new_dim_numbers.mutable_input_spatial_dimensions(),
                 input_shape);

  new_dim_numbers.set_kernel_input_feature_dimension(transpose_dim(
      dim_numbers.kernel_input_feature_dimension(), filter_shape));
  new_dim_numbers.set_kernel_output_feature_dimension(transpose_dim(
      dim_numbers.kernel_output_feature_dimension(), filter_shape));
  transpose_dims(*new_dim_numbers.mutable_kernel_spatial_dimensions(),
                 filter_shape);

  new_dim_numbers.set_output_batch_dimension(
      transpose_dim(dim_numbers.output_batch_dimension(), output_shape));
  new_dim_numbers.set_output_feature_dimension(
      transpose_dim(dim_numbers.output_feature_dimension(), output_shape));
  transpose_dims(*new_dim_numbers.mutable_output_spatial_dimensions(),
                 output_shape);

  Shape normalized_shape;
  if (hlo->shape().IsTuple()) {
    TF_RET_CHECK(hlo->shape().tuple_shapes_size() == 2);
    TF_RET_CHECK(hlo->shape().tuple_shapes(1).rank() == 1)
        << "Second element in a convolution tuple is expected to be an "
           "allocator of rank one";
    normalized_shape = ShapeUtil::MakeTupleShape(
        {ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
             hlo->shape().tuple_shapes(0)),
         hlo->shape().tuple_shapes(1)});
  } else {
    normalized_shape =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
            hlo->shape());
  }

  // We need to restore degenerate dimensions, since those might be used in
  // either batch dimension, or contracting dimensions.
  std::vector<HloInstruction*> normalized_operands;
  for (int idx = 0; idx < hlo->operand_count(); idx++) {
    HloInstruction* op = hlo->mutable_operand(idx);
    const Shape& s = op->shape();
    Shape s_reordered =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(s);
    HloInstruction* normalized_op = op->mutable_operand(0);
    HloInstruction* new_op;
    if (normalized_op->shape() == s_reordered) {
      new_op = normalized_op;
    } else {
      new_op = MakeBitcastHlo(op, s_reordered);
    }
    normalized_operands.push_back(new_op);
  }

  HloInstruction* normalized_conv = hlo->parent()->AddInstruction(
      HloInstruction::CreateCustomCall(normalized_shape, normalized_operands,
                                       hlo->custom_call_target()),
      &hlo->metadata());

  normalized_conv->set_window(hlo->window());
  normalized_conv->set_convolution_dimension_numbers(new_dim_numbers);
  normalized_conv->set_feature_group_count(hlo->feature_group_count());
  normalized_conv->set_raw_backend_config_string(
      hlo->raw_backend_config_string());
  normalized_conv->parent()->parent()->SetAndUniquifyInstrName(normalized_conv,
                                                               hlo->name());

  // We are hoping that AlgebraicSimplifier will simplify the extraneous
  // tuples built this way.
  HloInstruction* bc_to_orig;
  if (normalized_conv->shape().IsTuple()) {
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_out,
                        MakeGetTupleElementHlo(normalized_conv, 0));
    TF_ASSIGN_OR_RETURN(HloInstruction * allocator,
                        MakeGetTupleElementHlo(normalized_conv, 1));
    HloInstruction* orig_shape_out =
        MakeBitcastHlo(normalized_out, hlo->shape().tuple_shapes(0));
    bc_to_orig = MaybeMakeTuple({orig_shape_out, allocator});
  } else {
    bc_to_orig = MakeBitcastHlo(normalized_conv, hlo->shape());
  }

  return std::make_optional(bc_to_orig);
}

}  // end namespace gpu
}  // end namespace xla
