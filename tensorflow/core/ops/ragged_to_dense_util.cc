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

#include "tensorflow/core/ops/ragged_to_dense_util.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace tensorflow {

using errors::InvalidArgument;

string RowPartitionTypeToString(RowPartitionType row_partition_type) {
  switch (row_partition_type) {
    case RowPartitionType::FIRST_DIM_SIZE:
      return "FIRST_DIM_SIZE";
    case RowPartitionType::VALUE_ROWIDS:
      return "VALUE_ROWIDS";
    case RowPartitionType::ROW_LENGTHS:
      return "ROW_LENGTHS";
    case RowPartitionType::ROW_SPLITS:
      return "ROW_SPLITS";
    case RowPartitionType::ROW_LIMITS:
      return "ROW_LIMITS";
    case RowPartitionType::ROW_STARTS:
      return "ROW_STARTS";
    default:
      return "UNKNOWN ROW PARTITION TYPE";
  }
}
tensorflow::Status GetRowPartitionTypesHelper(
    const std::vector<string>& row_partition_type_strings,
    std::vector<RowPartitionType>* row_partition_types) {
  static const auto kStringToType =
      new std::unordered_map<string, RowPartitionType>(
          {{"FIRST_DIM_SIZE", RowPartitionType::FIRST_DIM_SIZE},
           {"VALUE_ROWIDS", RowPartitionType::VALUE_ROWIDS},
           {"ROW_LENGTHS", RowPartitionType::ROW_LENGTHS},
           {"ROW_SPLITS", RowPartitionType::ROW_SPLITS},
           {"ROW_LIMITS", RowPartitionType::ROW_LIMITS},
           {"ROW_STARTS", RowPartitionType::ROW_STARTS}});

  for (const string& type_str : row_partition_type_strings) {
    const auto iter = kStringToType->find(type_str);
    if (iter == kStringToType->end()) {
      return InvalidArgument("Unknown string for partition info type: ",
                             type_str);
    }
    row_partition_types->push_back(iter->second);
  }
  return tensorflow::Status::OK();
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
    return tensorflow::Status::OK();
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
    return tensorflow::Status::OK();
  }
  // At this point, value_shape and output_shape have known ranks.
  if (ragged_rank + value_shape.dim_size() != output_shape->dim_size()) {
    return InvalidArgument("Value shape (", value_shape.DebugString(),
                           "), ragged_rank(", ragged_rank, ") and shape(",
                           shape.DebugString(),
                           ") do not have a consistent number of dimensions");
  }

  for (int i = 1; i < value_shape.dim_size(); ++i) {
    const TensorShapeProto::Dim& value_dim = value_shape.dim(i);
    TensorShapeProto::Dim* output_shape_dim = output_shape->mutable_dim(
        output_shape->dim_size() - value_shape.dim_size() + i);

    if (value_dim.size() >= 0) {
      if (output_shape_dim->size() >= 0) {
        if (output_shape_dim->size() != value_dim.size()) {
          return InvalidArgument("Value and shape dimension are inconsistent.");
        }
      } else {
        output_shape_dim->set_size(value_dim.size());
      }
    }
  }
  return tensorflow::Status::OK();
}

int GetRaggedRank(const std::vector<RowPartitionType>& row_partition_types) {
  if (row_partition_types.empty()) {
    return 0;
  }
  if (row_partition_types[0] == RowPartitionType::FIRST_DIM_SIZE) {
    return row_partition_types.size() - 1;
  }
  return row_partition_types.size();
}

tensorflow::Status ValidateDefaultValueShape(
    const TensorShapeProto& default_value_shape,
    const TensorShapeProto& value_shape) {
  if (default_value_shape.unknown_rank() || value_shape.unknown_rank()) {
    return tensorflow::Status::OK();
  }

  if (default_value_shape.dim_size() > value_shape.dim_size()) {
    // TODO(martinz): This constraint is unnecessary. The
    // default value could have as many dimensions as shape. If there is a
    // discrepancy, it will be picked up when we broadcast the default value.
    // For now, I'll relax the constraint only slightly.
    return InvalidArgument(
        "default_value_shape must have no more dimensions than the value. "
        "default_value_shape: ",
        default_value_shape.DebugString(),
        " default_value_shape.dim_size(): ", default_value_shape.dim_size(),
        " value_shape: ", value_shape.DebugString(),
        " value_shape.dim_size(): ", value_shape.dim_size());
  }
  for (int i = 0;
       i < std::min(default_value_shape.dim_size(), value_shape.dim_size() - 1);
       ++i) {
    if (default_value_shape.dim(i).size() >= 0 &&
        value_shape.dim(i + 1).size() >= 0 &&
        default_value_shape.dim(i).size() != 1 &&
        default_value_shape.dim(i).size() != value_shape.dim(i + 1).size()) {
      return InvalidArgument(
          "default_value_shape and value_shape do not match on dimension ", i);
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace tensorflow
