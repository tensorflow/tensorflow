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

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "xla/hlo/builder/lib/broadcast.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

absl::StatusOr<xla::XlaOp> BroadcastTo(xla::XlaOp input,
                                       absl::Span<int64_t const> output_dims) {
  return xla::BroadcastTo(input, output_dims);
}

xla::XlaOp BroadcastDynamicDimensions(xla::XlaOp input,
                                      absl::Span<int64_t const> output_dims) {
  xla::XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape input_shape, builder->GetShape(input));
    const int64_t rank = input_shape.dimensions().size();
    const int64_t rank_difference =
        static_cast<int64_t>(output_dims.size()) - rank;
    // BroadcastTo reports an output of lower rank.
    for (int64_t i = 0; rank_difference >= 0 && i < rank; ++i) {
      const int64_t output_size = output_dims[i + rank_difference];
      if (input_shape.is_dynamic_dimension(i) &&
          input_shape.dimensions(i) > 1 && output_size > 1) {
        input = BroadcastDynamicDimension(input, i, output_size);
      }
    }
    return input;
  });
}

xla::XlaOp BroadcastDynamicDimension(xla::XlaOp input, int64_t dimension,
                                     int64_t size) {
  xla::XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(input));
    const int64_t bound = shape.dimensions(dimension);
    xla::XlaOp run_time_size = xla::GetDimensionSize(input, dimension);
    xla::XlaOp data = xla::RemoveDynamicDimension(input, dimension);
    if (bound > size) {
      data = xla::SliceInDim(data, 0, size, 1, dimension);
      run_time_size =
          xla::Min(run_time_size, xla::ScalarLike(run_time_size, size));
    } else if (bound < size) {
      data = xla::PadInDim(data, xla::Zero(builder, shape.element_type()),
                           dimension, 0, size - bound);
    }
    shape.set_dimensions(dimension, size);
    std::vector<int64_t> broadcast_dimensions(shape.dimensions().size());
    absl::c_iota(broadcast_dimensions, 0);
    xla::XlaOp first =
        xla::BroadcastInDim(xla::SliceInDim(data, 0, 1, 1, dimension),
                            shape.dimensions(), broadcast_dimensions);
    xla::XlaOp is_one =
        xla::Eq(run_time_size, xla::ScalarLike(run_time_size, 1));
    data = xla::Select(is_one, first, data);
    return xla::SetDimensionSize(
        data,
        xla::Select(is_one, xla::ScalarLike(run_time_size, size),
                    run_time_size),
        dimension);
  });
}

absl::Status BroadcastOpsToSame(xla::XlaOp* lhs, xla::XlaOp* rhs) {
  TF_ASSIGN_OR_RETURN(auto lhs_xla_shape, lhs->builder()->GetShape(*lhs));
  TF_ASSIGN_OR_RETURN(auto rhs_xla_shape, rhs->builder()->GetShape(*rhs));
  tensorflow::TensorShape lhs_tf_shape;
  tensorflow::TensorShape rhs_tf_shape;
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(lhs_xla_shape, &lhs_tf_shape));
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(rhs_xla_shape, &rhs_tf_shape));
  if (!lhs_tf_shape.IsSameSize(rhs_tf_shape)) {
    tensorflow::BCast bcast(tensorflow::BCast::FromShape(lhs_tf_shape),
                            tensorflow::BCast::FromShape(rhs_tf_shape));
    if (!bcast.IsValid()) {
      return absl::InvalidArgumentError(
          "Dimensions cannot be made to match through broadcasting");
    }
    TF_ASSIGN_OR_RETURN(*lhs, xla::BroadcastTo(*lhs, bcast.output_shape()));
    TF_ASSIGN_OR_RETURN(*rhs, xla::BroadcastTo(*rhs, bcast.output_shape()));
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
