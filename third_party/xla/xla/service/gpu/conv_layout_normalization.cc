/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/conv_layout_normalization.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

absl::StatusOr<std::optional<HloInstruction*>> UpdateLayoutForCudnnConvolution(
    HloCustomCallInstruction* hlo) {
  HloInstruction* lhs = hlo->mutable_operand(0);
  HloInstruction* rhs = hlo->mutable_operand(1);
  const ConvolutionDimensionNumbers& dim_numbers =
      hlo->convolution_dimension_numbers();

  auto transpose_dim = [&](int64_t dim, const Shape& unnormalized_shape) {
    return unnormalized_shape.dimensions().size() -
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
    case gpu::CudnnConvKind::kForwardActivation:
    case gpu::CudnnConvKind::kForwardGraph: {
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
    TF_RET_CHECK(hlo->shape().tuple_shapes().back().dimensions().size() == 1)
        << "The last element in the tuple returned by a convolution Custom "
           "Call is expected to be an "
           "allocator of rank one";
    std::vector<Shape> new_tuple_shape;
    for (const Shape& tuple_shape : hlo->shape().tuple_shapes()) {
      new_tuple_shape.emplace_back(
          ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
              tuple_shape));
    }
    normalized_shape = ShapeUtil::MakeTupleShape(new_tuple_shape);
  } else {
    normalized_shape =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
            hlo->shape());
  }

  // We need to restore degenerate dimensions, since those might be used in
  // either batch dimension, or contracting dimensions.
  std::vector<HloInstruction*> normalized_operands;
  bool performed_normalization = false;
  for (int idx = 0; idx < hlo->operand_count(); idx++) {
    HloInstruction* op = hlo->mutable_operand(idx);
    const Shape& s = op->shape();
    Shape s_reordered =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(s);
    normalized_operands.emplace_back(MakeBitcastHlo(op, s_reordered));
  }

  // Avoid replacing the Custom Call with an identical copy.
  if (!performed_normalization &&
      ShapeUtil::Equal(normalized_shape, hlo->shape()) &&
      ConvolutionDimensionNumbersToString(new_dim_numbers) ==
          ConvolutionDimensionNumbersToString(dim_numbers)) {
    return std::nullopt;
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
  *normalized_conv->mutable_precision_config() = hlo->precision_config();
  normalized_conv->parent()->parent()->SetAndUniquifyInstrName(normalized_conv,
                                                               hlo->name());

  // We are hoping that AlgebraicSimplifier will simplify the extraneous
  // tuples built this way.
  HloInstruction* bc_to_orig;
  if (normalized_conv->shape().IsTuple()) {
    std::vector<HloInstruction*> tuple_elements(
        normalized_conv->shape().tuple_shapes_size());

    for (int i = 0; i < normalized_conv->shape().tuple_shapes_size(); ++i) {
      TF_ASSIGN_OR_RETURN(HloInstruction * normalized_out,
                          MakeGetTupleElementHlo(normalized_conv, i));
      tuple_elements[i] =
          MakeBitcastHlo(normalized_out, hlo->shape().tuple_shapes(i));
    }
    bc_to_orig = MaybeMakeTuple(tuple_elements);
  } else {
    bc_to_orig = MakeBitcastHlo(normalized_conv, hlo->shape());
  }
  return bc_to_orig;
}

}  // namespace

absl::StatusOr<std::optional<HloInstruction*>> NormalizeLayoutForGpuCustomCalls(
    HloCustomCallInstruction* hlo) {
  if (IsCustomCallToDnnConvolution(*hlo)) {
    TF_ASSIGN_OR_RETURN(std::optional<HloInstruction*> bc_to_orig,
                        UpdateLayoutForCudnnConvolution(hlo));
    return bc_to_orig;
  }
  return std::nullopt;
}

}  // end namespace gpu
}  // end namespace xla
