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

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

absl::Status RaggedGatherShapeFn(InferenceContext* c);

//==============================================================================
// Registered Ops
//==============================================================================

REGISTER_OP("RaggedGather")
    .Input("params_nested_splits: PARAMS_RAGGED_RANK * Tsplits")
    .Input("params_dense_values: Tvalues")
    .Input("indices: Tindices")
    .Output("output_nested_splits: OUTPUT_RAGGED_RANK * Tsplits")
    .Output("output_dense_values: Tvalues")
    .Attr("Tvalues: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .Attr("PARAMS_RAGGED_RANK: int >= 1")
    .Attr("OUTPUT_RAGGED_RANK: int >= 0")
    .SetShapeFn(RaggedGatherShapeFn);

REGISTER_OP("RaggedCross")
    .Input("ragged_values: ragged_values_types")
    .Input("ragged_row_splits: ragged_splits_types")
    .Input("sparse_indices: Nsparse * int64")
    .Input("sparse_values: sparse_values_types")
    .Input("sparse_shape: Nsparse * int64")
    .Input("dense_inputs: dense_types")
    .Output("output_values: out_values_type")
    .Output("output_row_splits: out_row_splits_type")
    .Attr("Nsparse: int >= 0")
    .Attr("input_order: string")
    .Attr("hashed_output: bool")
    .Attr("num_buckets: int >= 0")
    .Attr("hash_key: int")
    .Attr("ragged_values_types: list({int64, string}) >= 0")
    .Attr("ragged_splits_types: list({int32, int64}) >= 0")
    .Attr("sparse_values_types: list({int64, string}) >= 0")
    .Attr("dense_types: list({int64, string}) >= 0")
    .Attr("out_values_type: {int64, string}")
    .Attr("out_row_splits_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<DataType> ragged_values_types;
      std::vector<DataType> ragged_splits_types;
      std::vector<DataType> sparse_values_types;
      std::vector<DataType> dense_types;

      TF_RETURN_IF_ERROR(
          c->GetAttr("ragged_values_types", &ragged_values_types));
      TF_RETURN_IF_ERROR(
          c->GetAttr("ragged_splits_types", &ragged_splits_types));
      TF_RETURN_IF_ERROR(c->GetAttr("dense_types", &dense_types));
      TF_RETURN_IF_ERROR(
          c->GetAttr("sparse_values_types", &sparse_values_types));

      int num_ragged = ragged_values_types.size();
      if (num_ragged != ragged_splits_types.size()) {
        return errors::InvalidArgument(
            "ragged values and splits must have the same length.");
      }

      int num_sparse;
      TF_RETURN_IF_ERROR(c->GetAttr("Nsparse", &num_sparse));
      if (num_sparse != sparse_values_types.size()) {
        return errors::InvalidArgument(
            "sparse indices and values must have the same length");
      }

      ShapeHandle out_values = c->UnknownShapeOfRank(1);
      ShapeHandle out_splits = c->UnknownShapeOfRank(1);

      // Merge the shapes of row_splits from ragged inputs.  (This is one plus
      // the batch size.)
      int ragged_splits_start = num_ragged;
      for (int i = 0; i < ragged_splits_types.size(); ++i) {
        ShapeHandle row_splits = c->input(i + ragged_splits_start);
        if (!c->Merge(out_splits, row_splits, &out_splits).ok()) {
          return errors::InvalidArgument(
              "inputs must all have the same batch dimension size.");
        }
      }

      // Merge the batch size of each dense input into out_splits.
      int dense_start = num_ragged * 2 + num_sparse * 3;
      for (int i = 0; i < dense_types.size(); ++i) {
        ShapeHandle dense_input = c->input(i + dense_start);
        int32 rank = c->Rank(dense_input);
        if (rank == InferenceContext::kUnknownRank) {
          continue;
        } else if (rank != 2) {
          return errors::InvalidArgument(
              "tf.ragged.cross only supports inputs with rank=2");
        }
        int64_t batch_size = c->Value(c->Dim(dense_input, 0));
        if (batch_size != InferenceContext::kUnknownDim) {
          ShapeHandle row_splits = c->Vector(batch_size + 1);
          if (!c->Merge(out_splits, row_splits, &out_splits).ok()) {
            return errors::InvalidArgument(
                "inputs must all have the same batch dimension size.");
          }
        }
      }

      c->set_output(0, out_values);
      c->set_output(1, out_splits);
      return absl::OkStatus();
    });

REGISTER_OP("RaggedFillEmptyRows")
    .Input("value_rowids: int64")
    .Input("values: T")
    .Input("nrows: int64")
    .Input("default_value: T")
    .Output("output_value_rowids: int64")
    .Output("output_values: T")
    .Output("empty_row_indicator: bool")
    .Output("reverse_index_map: int64")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_value_rowids = c->input(0);
      TF_RETURN_IF_ERROR(
          c->WithRank(input_value_rowids, 1, &input_value_rowids));
      ShapeHandle input_values = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(input_values, 1, &input_values));
      ShapeHandle input_nrows = c->input(2);
      TF_RETURN_IF_ERROR(c->WithRank(input_nrows, 0, &input_nrows));
      ShapeHandle default_value = c->input(3);
      TF_RETURN_IF_ERROR(c->WithRank(default_value, 0, &default_value));
      DimensionHandle N = c->Dim(input_value_rowids, 0);
      TF_RETURN_IF_ERROR(c->Merge(N, c->Dim(input_values, 0), &N));
      DimensionHandle unused_dim;
      ShapeHandle output_value_rowids =
          c->Vector(InferenceContext::kUnknownDim);
      ShapeHandle output_values = c->Vector(InferenceContext::kUnknownDim);
      ShapeHandle empty_row_indicator =
          c->Vector(InferenceContext::kUnknownDim);
      ShapeHandle reverse_index_map = c->Vector(N);
      c->set_output(0, output_value_rowids);
      c->set_output(1, output_values);
      c->set_output(2, empty_row_indicator);
      c->set_output(3, reverse_index_map);
      return absl::OkStatus();
    });

REGISTER_OP("RaggedFillEmptyRowsGrad")
    .Input("reverse_index_map: int64")
    .Input("grad_values: T")
    .Output("d_values: T")
    .Output("d_default_value: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle reverse_index_map = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(reverse_index_map, 1, &reverse_index_map));
      ShapeHandle grad_values = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(grad_values, 1, &grad_values));
      c->set_output(0, reverse_index_map);
      c->set_output(1, c->Scalar());
      return absl::OkStatus();
    });

//==============================================================================
// Shape Functions
//==============================================================================

absl::Status RaggedGatherShapeFn(InferenceContext* c) {
  int num_splits;
  int64_t PARAMS_RAGGED_RANK;
  TF_RETURN_IF_ERROR(
      c->GetAttr<int64_t>("PARAMS_RAGGED_RANK", &PARAMS_RAGGED_RANK));
  TF_RETURN_IF_ERROR(c->GetAttr<int>("OUTPUT_RAGGED_RANK", &num_splits));

  // Check rank of `indices`.
  ShapeHandle indices = c->input(PARAMS_RAGGED_RANK + 1);
  TF_RETURN_IF_ERROR(
      c->WithRank(indices, num_splits - PARAMS_RAGGED_RANK + 1, &indices));

  // Check that all params_nested_splits have rank 1.
  for (int64_t i = 0; i < PARAMS_RAGGED_RANK; ++i) {
    ShapeHandle splits = c->input(i);
    TF_RETURN_IF_ERROR(c->WithRank(splits, 1, &splits));
  }

  // Check that `params_dense_values` has rank>=1.
  ShapeHandle params_dense_values = c->input(PARAMS_RAGGED_RANK);
  TF_RETURN_IF_ERROR(
      c->WithRankAtLeast(params_dense_values, 1, &params_dense_values));

  // Set the rank for the `splits` outputs.
  for (int i = 0; i < num_splits; ++i) {
    c->set_output(i, c->UnknownShapeOfRank(1));
  }

  // Calculate the `values` shape.
  ShapeHandle value = c->UnknownShape();
  ShapeHandle values = c->UnknownShape();
  TF_RETURN_IF_ERROR(c->Subshape(params_dense_values, 1, &value));
  TF_RETURN_IF_ERROR(c->Concatenate(c->UnknownShapeOfRank(1), value, &values));
  c->set_output(num_splits, values);

  return absl::OkStatus();
}

}  // namespace tensorflow
