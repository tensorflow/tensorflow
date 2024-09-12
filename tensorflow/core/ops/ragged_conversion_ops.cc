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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/ragged_to_dense_util.h"

namespace tensorflow {

using errors::InvalidArgument;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {
tensorflow::Status ValidateRowPartitionTypesAndShapes(
    const std::vector<RowPartitionType>& row_partition_types,
    InferenceContext* c) {
  // Note: the allowed types may be extended in the future.
  for (RowPartitionType row_partition_type : row_partition_types) {
    switch (row_partition_type) {
      case RowPartitionType::FIRST_DIM_SIZE:
      case RowPartitionType::VALUE_ROWIDS:
      case RowPartitionType::ROW_SPLITS:
        break;
      default:
        return InvalidArgument("Unsupported partition type: ",
                               RowPartitionTypeToString(row_partition_type));
    }
  }

  if (row_partition_types.empty()) {
    return InvalidArgument("Partition info types should not be empty");
  }
  for (int i = 1; i < row_partition_types.size(); ++i) {
    if (row_partition_types[i] == RowPartitionType::FIRST_DIM_SIZE) {
      return InvalidArgument("FIRST_DIM_SIZE must be first");
    }
  }
  if (row_partition_types[0] == RowPartitionType::FIRST_DIM_SIZE &&
      (row_partition_types.size() < 2 ||
       row_partition_types[1] != RowPartitionType::VALUE_ROWIDS)) {
    return InvalidArgument("FIRST_DIM_SIZE must be followed by VALUE_ROWIDS");
  }
  if (row_partition_types[0] == RowPartitionType::VALUE_ROWIDS) {
    return InvalidArgument("VALUE_ROWIDS cannot be first");
  }

  int num_row_partition_tensors;
  TF_RETURN_IF_ERROR(
      c->GetAttr("num_row_partition_tensors", &num_row_partition_tensors));
  if (num_row_partition_tensors != row_partition_types.size()) {
    return InvalidArgument(
        "Number of row partition tensors (", num_row_partition_tensors,
        ") does not equal the number of row partition types(",
        row_partition_types.size(), ").");
  }

  for (int i = 0; i < num_row_partition_tensors; ++i) {
    TensorShapeProto partition_shape;
    c->ShapeHandleToProto(c->input(3 + i), &partition_shape);
    if (partition_shape.unknown_rank()) {
      continue;
    }
    if (row_partition_types[i] == RowPartitionType::FIRST_DIM_SIZE) {
      if (partition_shape.dim_size() != 0) {
        return InvalidArgument("FIRST_DIM_SIZE must be a scalar.");
      }
    } else {
      if (partition_shape.dim_size() != 1) {
        return InvalidArgument("Row partition must be a vector.");
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

Status RaggedTensorToSparseShapeFn(InferenceContext* c);
Status RaggedTensorToVariantShapeFn(InferenceContext* c);
Status RaggedTensorFromVariantShapeFn(InferenceContext* c);
Status RaggedTensorToVariantGradientShapeFn(InferenceContext* c);
Status RaggedTensorToTensorShapeFn(InferenceContext* c);

//==============================================================================
// Registered Ops
//==============================================================================

REGISTER_OP("RaggedTensorToSparse")
    .Input("rt_nested_splits: RAGGED_RANK * Tsplits")
    .Input("rt_dense_values: T")
    .Output("sparse_indices: int64")
    .Output("sparse_values: T")
    .Output("sparse_dense_shape: int64")
    .Attr("RAGGED_RANK: int >= 1")
    .Attr("T: type")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .SetShapeFn(RaggedTensorToSparseShapeFn);

REGISTER_OP("RaggedTensorToVariant")
    .Input("rt_nested_splits: RAGGED_RANK * Tsplits")
    .Input("rt_dense_values: Tvalues")
    .Output("encoded_ragged: variant")
    .Attr("RAGGED_RANK: int >= 0")
    .Attr("Tvalues: type")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .Attr("batched_input: bool")
    .SetTypeConstructor(full_type::Unary(TFT_RAGGED, "Tvalues"))
    .SetShapeFn(RaggedTensorToVariantShapeFn);

REGISTER_OP("RaggedTensorFromVariant")
    .Input("encoded_ragged: variant")
    .Output("output_nested_splits: output_ragged_rank * Tsplits")
    .Output("output_dense_values: Tvalues")
    .Attr("input_ragged_rank: int >= -1")
    .Attr("output_ragged_rank: int >= 0")
    .Attr("Tvalues: type")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .SetShapeFn(RaggedTensorFromVariantShapeFn);

REGISTER_OP("RaggedTensorToVariantGradient")
    .Input("encoded_ragged_grad: variant")
    .Input("row_splits: Tsplits")
    .Input("dense_values_shape: int32")
    .Output("dense_values_grad: Tvalues")
    .Attr("Tvalues: type")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .SetShapeFn(RaggedTensorToVariantGradientShapeFn);

REGISTER_OP("RaggedTensorToTensor")
    .Attr("T: type")
    .Attr("Tindex: {int64, int32}")
    .Attr("Tshape: {int64, int32}")
    .Attr("num_row_partition_tensors: int")
    .Attr("row_partition_types: list(string)")
    .Input("shape: Tshape")
    .Input("values: T")
    .Input("default_value: T")
    .Input("row_partition_tensors: num_row_partition_tensors * Tindex")
    .Output("result: T")
    .SetShapeFn(RaggedTensorToTensorShapeFn);

//==============================================================================
// Shape Functions
//==============================================================================

Status RaggedTensorToSparseShapeFn(InferenceContext* c) {
  int64_t num_splits;
  TF_RETURN_IF_ERROR(c->GetAttr<int64_t>("RAGGED_RANK", &num_splits));
  // TODO(b/112274756): Allow ragged_rank to be 0.
  if (num_splits < 1) {
    return errors::InvalidArgument("Requires RAGGED_RANK>0");
  }
  ShapeHandle rt_dense_values = c->input(num_splits);
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(rt_dense_values, 1, &rt_dense_values));

  // Check that all rt_nested_splits have rank 1.
  for (int64_t i = 0; i < num_splits; ++i) {
    ShapeHandle splits = c->input(i);
    TF_RETURN_IF_ERROR(c->WithRank(splits, 1, &splits));
  }

  DimensionHandle dense_dims =
      c->RankKnown(rt_dense_values)
          ? c->MakeDim(c->Rank(rt_dense_values) + num_splits)
          : c->UnknownDim();
  DimensionHandle num_values = c->NumElements(rt_dense_values);

  c->set_output(0, c->Matrix(num_values, dense_dims));  // indices
  c->set_output(1, c->Vector(num_values));              // values
  c->set_output(2, c->Vector(dense_dims));              // dense_shape

  return absl::OkStatus();
}

Status RaggedTensorToVariantShapeFn(InferenceContext* c) {
  int64_t num_splits;
  TF_RETURN_IF_ERROR(c->GetAttr<int64_t>("RAGGED_RANK", &num_splits));
  bool batched;
  TF_RETURN_IF_ERROR(c->GetAttr<bool>("batched_input", &batched));
  shape_inference::ShapeHandle rt_dense_values = c->input(num_splits);
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(rt_dense_values, 1, &rt_dense_values));
  for (int64_t i = 0; i < num_splits; ++i) {
    shape_inference::ShapeHandle splits = c->input(i);
    TF_RETURN_IF_ERROR(c->WithRank(splits, 1, &splits));
  }
  if (batched) {
    auto num_first_splits = c->Dim(c->input(0), 0);
    shape_inference::DimensionHandle num_rows;
    TF_RETURN_IF_ERROR(c->Subtract(num_first_splits, 1, &num_rows));
    c->set_output(0, c->Vector(num_rows));
  } else {
    c->set_output(0, c->Scalar());
  }
  return absl::OkStatus();
}

Status RaggedTensorToVariantGradientShapeFn(InferenceContext* c) {
  ShapeHandle shape;
  TF_RETURN_IF_ERROR(
      c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(2, &shape));
  c->set_output(0, shape);
  return absl::OkStatus();
}

Status RaggedTensorFromVariantShapeFn(InferenceContext* c) {
  int64_t input_ragged_rank;
  TF_RETURN_IF_ERROR(
      c->GetAttr<int64_t>("input_ragged_rank", &input_ragged_rank));
  int64_t output_ragged_rank;
  TF_RETURN_IF_ERROR(
      c->GetAttr<int64_t>("output_ragged_rank", &output_ragged_rank));
  shape_inference::ShapeHandle encoded_ragged = c->input(0);
  if (c->RankKnown(encoded_ragged) && input_ragged_rank >= 0) {
    shape_inference::ShapeHandle unused;
    TF_RETURN_IF_ERROR(c->WithRank(
        encoded_ragged, output_ragged_rank - input_ragged_rank, &unused));
  }
  for (int64_t i = 0; i < output_ragged_rank; i++) {
    c->set_output(i, c->UnknownShapeOfRank(1));
  }
  c->set_output(output_ragged_rank, c->UnknownShape());
  return absl::OkStatus();
}

tensorflow::Status RaggedTensorToTensorShapeFn(InferenceContext* c) {
  TensorShapeProto shape;
  {
    ShapeHandle shape_handle;
    TF_RETURN_IF_ERROR(
        c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(0, &shape_handle));
    c->ShapeHandleToProto(shape_handle, &shape);
  }

  std::vector<RowPartitionType> row_partition_types;
  TF_RETURN_IF_ERROR(GetRowPartitionTypes(c, &row_partition_types));
  int ragged_rank = GetRaggedRank(row_partition_types);
  TF_RETURN_IF_ERROR(
      ValidateRowPartitionTypesAndShapes(row_partition_types, c));

  TensorShapeProto value_shape;
  c->ShapeHandleToProto(c->input(1), &value_shape);

  TensorShapeProto default_value_shape;
  c->ShapeHandleToProto(c->input(2), &default_value_shape);

  TF_RETURN_IF_ERROR(
      ValidateDefaultValueShape(default_value_shape, value_shape));

  // TODO(martinz): Theoretically, we could check the first dimension of
  // value_shape against the first dimension of the last row_partition_tensor
  // assuming it is a VALUE_ROWIDS type.
  // TODO(martinz): Although we normally don't know the first dimension of the
  // output, we could infer it from the first dimension of the first
  // row_partition_tensor if it is ROW_SPLITS type.
  // TODO(martinz): If the shape is provided, but the value_shape has missing
  // dimensions, we can check the default_value_shape against the shape.
  TensorShapeProto output_shape;
  TF_RETURN_IF_ERROR(CombineRaggedTensorToTensorShapes(
      ragged_rank, shape, value_shape, &output_shape));

  ShapeHandle output_shape_handle;
  TF_RETURN_IF_ERROR(
      c->MakeShapeFromShapeProto(output_shape, &output_shape_handle));
  c->set_output(0, output_shape_handle);
  return absl::OkStatus();
}

}  // namespace tensorflow
