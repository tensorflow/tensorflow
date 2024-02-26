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

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "xla/client/lib/broadcast.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

absl::StatusOr<xla::XlaOp> BroadcastTo(xla::XlaOp input,
                                       absl::Span<int64_t const> output_dims) {
  return xla::BroadcastTo(input, output_dims);
}

Status BroadcastOpsToSame(xla::XlaOp* lhs, xla::XlaOp* rhs) {
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
      return tensorflow::errors::InvalidArgument(
          "Dimensions cannot be made to match through broadcasting");
    }
    TF_ASSIGN_OR_RETURN(*lhs, xla::BroadcastTo(*lhs, bcast.output_shape()));
    TF_ASSIGN_OR_RETURN(*rhs, xla::BroadcastTo(*rhs, bcast.output_shape()));
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
