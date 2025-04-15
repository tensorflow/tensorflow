/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("TPUPartitionedOutput")
    .Input("inputs:  T")
    .Output("output: num_splits * T")
    .Attr("T: type")
    .Attr("num_splits: int >= 1")
    .Attr("partition_dim: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &dtype));
      int partition_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("partition_dim", &partition_dim));
      int num_splits;
      TF_RETURN_IF_ERROR(c->GetAttr("num_splits", &num_splits));
      if (dtype == DT_RESOURCE) {
        return absl::UnimplementedError("Not implemented.");
      } else if (c->num_inputs() == 0) {
        return absl::InvalidArgumentError(
            "Expected at least one input to TPUPartitionedOutput.");
      }

      ShapeHandle input = c->input(0);
      int rank = InferenceContext::Rank(input);
      // limitation: can only validate rank when it is known
      if ((rank != InferenceContext::kUnknownRank && partition_dim >= rank) ||
          (partition_dim < -1))
        return absl::InvalidArgumentError(
            absl::StrCat("Cannot partition dim ", partition_dim, " of rank ",
                         rank, " tensor."));

      ShapeHandle newoutput0;
      if (partition_dim == -1) {
        newoutput0 = input;  // replicated input/output share shapes
      } else {
        shape_inference::DimensionHandle new_dim;
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            c->Divide(c->Dim(input, partition_dim), num_splits,
                      true /* evenly_divisible */, &new_dim),
            "Number of ways to split should evenly divide the split dimension");
        TF_CHECK_OK(c->ReplaceDim(input, partition_dim, new_dim, &newoutput0));
      }

      for (int i = num_splits - 1; i >= 0; --i) {
        c->set_output(i, newoutput0);
      }

      return absl::OkStatus();
    });

REGISTER_OP("TPUPartitionedOutputV2")
    .Input("inputs:  T")
    .Output("output: num_splits * T")
    .Attr("T: type")
    .Attr("num_splits: int >= 1")
    .Attr("partition_dims: list(int)")
    .SetShapeFn([](InferenceContext* c) {
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &dtype));
      std::vector<int> partition_dims;
      TF_RETURN_IF_ERROR(c->GetAttr("partition_dims", &partition_dims));
      int num_splits;
      TF_RETURN_IF_ERROR(c->GetAttr("num_splits", &num_splits));
      if (dtype == DT_RESOURCE) {
        return absl::UnimplementedError("Not implemented.");
      } else if (c->num_inputs() == 0) {
        return absl::InvalidArgumentError(
            "Expected at least one input to TPUPartitionedOutputV2.");
      }

      ShapeHandle handle = c->input(0);
      int rank = InferenceContext::Rank(handle);
      int num_cores_per_replica = 1;
      for (const int& partition_dim : partition_dims) {
        num_cores_per_replica *= partition_dim;
      }

      if (num_splits != num_cores_per_replica) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected ", num_cores_per_replica, " splits."));
      } else if (rank > (int)partition_dims.size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected at least ", rank, " partition dimensions."));
      }

      for (int i = 0; i < rank; ++i) {
        shape_inference::DimensionHandle dim;
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            c->Divide(c->Dim(handle, i), partition_dims[i],
                      true /* evenly_divisible */, &dim),
            "Number of ways to split should evenly divide the split dimension");
        TF_CHECK_OK(c->ReplaceDim(handle, i, dim, &handle));
      }

      for (int i = num_splits - 1; i >= 0; --i) {
        c->set_output(i, handle);
      }

      return absl::OkStatus();
    });

}  // namespace tensorflow
