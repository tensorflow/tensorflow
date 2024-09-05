/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/ragged_to_dense_util.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace tensorflow {

using errors::InvalidArgument;

tensorflow::Status GetRowPartitionTypesHelper(
    const std::vector<string>& row_partition_type_strings,
    std::vector<RowPartitionType>* row_partition_types) {
  *row_partition_types = GetRowPartitionTypesHelper(row_partition_type_strings);
  if (row_partition_types->size() != row_partition_type_strings.size()) {
    // Something was not converted, return error status.
    return InvalidArgument(
        "Unknown string for partition info type: ",
        row_partition_type_strings.at(row_partition_types->size()));
  }
  return absl::OkStatus();
}

tensorflow::Status CombineRaggedTensorToTensorShapes(
    int ragged_rank, const TensorShapeProto& shape,
    const TensorShapeProto& value_shape, TensorShapeProto* output_shape) {
  // Test for consistency of value_shape and shape specified.
  // If shape is unspecified and value_shape is specified, then copy
  // over the size from the value_shape dimension.

  if (value_shape.unknown_rank() && shape.unknown_rank()) {
    output_shape->Clear();
    output_shape->set_unknown_rank(true);
    return absl::OkStatus();
  }

  if (shape.unknown_rank()) {
    // Here, value_shape must be of known size.
    while (output_shape->dim_size() < ragged_rank + value_shape.dim_size()) {
      output_shape->add_dim()->set_size(-1);
    }
  } else {
    *output_shape = shape;
  }
  if (value_shape.unknown_rank()) {
    return absl::OkStatus();
  }
  // At this point, value_shape and output_shape have known ranks.
  if (ragged_rank + value_shape.dim_size() != output_shape->dim_size()) {
    return InvalidArgument(
        "rt_input.shape and shape=", TensorShape::DebugString(shape),
        " are incompatible: rt_input.rank = ",
        ragged_rank + value_shape.dim_size(),
        " but shape.rank = ", output_shape->dim_size());
  }

  for (int i = 1; i < value_shape.dim_size(); ++i) {
    const TensorShapeProto::Dim& value_dim = value_shape.dim(i);
    TensorShapeProto::Dim* output_shape_dim = output_shape->mutable_dim(
        output_shape->dim_size() - value_shape.dim_size() + i);

    if (value_dim.size() >= 0) {
      if (output_shape_dim->size() >= 0) {
        if (output_shape_dim->size() != value_dim.size()) {
          return InvalidArgument(
              "rt_input.shape and shape=", TensorShape::DebugString(shape),
              " are incompatible: rt_input.shape[", i + ragged_rank,
              "] = ", value_dim.size(), " but shape[", i + ragged_rank,
              "] = ", output_shape_dim->size());
        }
      } else {
        output_shape_dim->set_size(value_dim.size());
      }
    }
  }
  return absl::OkStatus();
}

tensorflow::Status ValidateDefaultValueShape(
    const TensorShapeProto& default_value_shape,
    const TensorShapeProto& value_shape) {
  if (default_value_shape.unknown_rank() || value_shape.unknown_rank()) {
    return absl::OkStatus();
  }

  int default_ndims = default_value_shape.dim_size();
  int values_ndims = value_shape.dim_size();
  if (default_ndims >= values_ndims) {
    return InvalidArgument(
        "default_value.shape=", TensorShape::DebugString(default_value_shape),
        " and rt_input.flat_values.shape=",
        TensorShape::DebugString(value_shape),
        " are incompatible: default_value.rank = ", default_ndims,
        "  must be less than rt_input.flat_values.rank = ", values_ndims);
  }
  for (int i = 0; i < std::min(default_ndims, values_ndims - 1); ++i) {
    int default_dim = default_value_shape.dim(i).size();
    int value_dim = value_shape.dim(i + 1).size();
    if (default_dim >= 0 && value_dim >= 0 && default_dim != 1 &&
        default_dim != value_dim) {
      return InvalidArgument(
          "default_value.shape=", TensorShape::DebugString(default_value_shape),
          " and rt_input.flat_values.shape=",
          TensorShape::DebugString(value_shape),
          " are incompatible: default_value.shape[",
          i - default_value_shape.dim_size(), "] = ", default_dim,
          " but rt_input.flat_values.shape[",
          i - default_value_shape.dim_size(), "] = ", value_dim);
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
