/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

namespace {

absl::Status ValidateSparseDenseMatmulCustomCombinerGradWithCsrInputShape(
    shape_inference::InferenceContext* c, const int weights_index,
    const int preserved_valencies_index, const int preserved_vectors_index,
    const int preserved_weights_index, const int activation_gradients_index,
    const int tables_index, const int num_tables) {
  shape_inference::ShapeHandle shape;
  int num_weights;
  int max_valency_int;
  TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));
  TF_RETURN_IF_ERROR(c->GetAttr("max_valency", &max_valency_int));
  // Only check the shape of the weights when num_weights > 0 to avoid
  // issues of 0-shaped values.
  if (num_weights > 0) {
    TF_RETURN_IF_ERROR(c->Merge(c->input(weights_index),
                                c->MakeShape({c->MakeDim(num_weights)}),
                                &shape));
    TF_RETURN_IF_ERROR(c->Merge(c->input(preserved_weights_index),
                                c->MakeShape({c->MakeDim(num_weights)}),
                                &shape));
  }
  // Check that the preserved tensors have the expected shapes:
  // valencies: [input_size];
  // vectors: [input_size, max_valency, feature_width];
  auto input_size = c->Dim(c->input(activation_gradients_index), 0);
  auto max_valency = c->MakeDim(max_valency_int);
  auto feature_width = c->Dim(c->input(tables_index), 1);
  TF_RETURN_IF_ERROR(c->Merge(c->input(preserved_valencies_index),
                              c->MakeShape({input_size}), &shape));
  TF_RETURN_IF_ERROR(
      c->Merge(c->input(preserved_vectors_index),
               c->MakeShape({input_size, max_valency, feature_width}), &shape));
  // `updated_tables` refers to both the embedding table and the associated
  // slot variables. They all have the same embedding table shape.
  for (int i = 0; i < num_tables; ++i) {
    c->set_output(i, c->input(tables_index));
  }
  // `updated_weights` simply have a 1D shape of `num_weights`.
  c->set_output(num_tables, c->MakeShape({c->MakeDim(num_weights)}));
  return absl::OkStatus();
}

}  // namespace

REGISTER_OP("XlaSparseDenseMatmul")
    .Input("row_ids: int32")
    .Input("col_ids: uint32")
    .Input("values: float32")
    .Input("offsets: uint32")
    .Input("embedding_table: float32")
    .Output("activations: float32")
    .Output("row_pointers: int32")
    .Output("sorted_embedding_ids: int32")
    .Output("sorted_sample_ids: int32")
    .Output("sorted_gains: float32")
    .Attr("max_ids_per_partition: int >= 0")
    .Attr("max_unique_ids_per_partition: int >= 0")
    .Attr("input_size: int >= 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int input_size;
      TF_RETURN_IF_ERROR(c->GetAttr("input_size", &input_size));
      shape_inference::ShapeHandle rank;
      for (int i = 0; i < 3; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &rank));
      }
      // TODO(bfontain): Should be a rank 1 tensor.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &rank));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &rank));
      for (int i = 1; i < 3; ++i) {
        shape_inference::ShapeHandle merged;
        TF_RETURN_IF_ERROR(c->Merge(c->input(i), c->input(0), &merged));
      }
      shape_inference::ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(4), 0, c->MakeDim(input_size), &output_shape));
      c->set_output(0, output_shape);
      // TODO(pineapplejuice233): Change this to concrete shape once SparseTensor is
      // available.
      c->set_output(1, c->UnknownShapeOfRank(1));
      c->set_output(2, c->UnknownShapeOfRank(1));
      c->set_output(3, c->UnknownShapeOfRank(1));
      c->set_output(4, c->UnknownShapeOfRank(1));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulWithCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("embedding_table: T")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("activations: T")
    .Attr("input_size: int >= 0")
    .Attr("quantization_config_low: float")
    .Attr("quantization_config_high: float")
    .Attr("quantization_config_num_buckets: int >= 0")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .Attr("T: {int32, float32} = DT_FLOAT")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int input_size;
      TF_RETURN_IF_ERROR(c->GetAttr("input_size", &input_size));
      shape_inference::ShapeHandle rank;
      for (int i = 0; i < 4; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &rank));
      }
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &rank));
      for (int i = 2; i < 4; ++i) {
        shape_inference::ShapeHandle merged;
        TF_RETURN_IF_ERROR(c->Merge(c->input(i), c->input(1), &merged));
      }
      shape_inference::ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(4), 0, c->MakeDim(input_size), &output_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &rank));
      c->set_output(0, output_shape);
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_pos_ids: int32")
    .Input("sorted_gains: float32")
    .Input("embedding_table: float32")
    .Input("weights: float32")
    .Output("activations: float32")
    .Output("preserved_valencies: int32")
    .Output("preserved_vectors: float32")
    .Attr("input_size: int >= 0")
    .Attr("max_valency: int >= 0")
    .Attr("num_weights: int >= 0")
    .Attr("combiner_computation: func")
    .Attr("quantization_config_low: float")
    .Attr("quantization_config_high: float")
    .Attr("quantization_config_num_buckets: int >= 0")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      constexpr int kRowPointersIndex = 0;
      constexpr int kSortedSampleIdsIndex = 1;
      constexpr int kEmbeddingTableIndex = 5;
      constexpr int kEmbeddingTableRank = 2;
      constexpr int kWeightsIndex = 6;
      constexpr int kWeightsRank = 1;
      constexpr int kOutputActivationsIndex = 0;
      constexpr int kPreservedValenciesIndex = 1;
      constexpr int kPreservedVectorsIndex = 2;
      // This input_size is per-chip batch size.
      int input_size;
      TF_RETURN_IF_ERROR(c->GetAttr("input_size", &input_size));
      int max_valency;
      TF_RETURN_IF_ERROR(c->GetAttr("max_valency", &max_valency));
      int num_weights;
      TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));

      shape_inference::ShapeHandle rank;
      for (int i = kRowPointersIndex; i < kEmbeddingTableIndex; ++i) {
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(i), kSortedSampleIdsIndex, &rank));
      }
      TF_RETURN_IF_ERROR(c->WithRank(c->input(kEmbeddingTableIndex),
                                     kEmbeddingTableRank, &rank));
      for (int i = kSortedSampleIdsIndex + 1; i < kEmbeddingTableIndex; ++i) {
        shape_inference::ShapeHandle merged;
        TF_RETURN_IF_ERROR(
            c->Merge(c->input(i), c->input(kSortedSampleIdsIndex), &merged));
      }
      if (num_weights > 0) {
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(kWeightsIndex), kWeightsRank, &rank));
        shape_inference::DimensionHandle weights_dim;
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(kWeightsIndex), 0),
                                        num_weights, &weights_dim));
      }

      shape_inference::DimensionHandle input_size_dim = c->MakeDim(input_size);
      shape_inference::DimensionHandle max_valency_dim =
          c->MakeDim(max_valency);
      shape_inference::DimensionHandle feature_width_dim =
          c->Dim(c->input(kEmbeddingTableIndex), 1);
      shape_inference::ShapeHandle output_activations_shape;
      TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(kEmbeddingTableIndex), 0,
                                       c->MakeDim(input_size),
                                       &output_activations_shape));
      c->set_output(kOutputActivationsIndex, output_activations_shape);
      c->set_output(kPreservedValenciesIndex, c->MakeShape({input_size_dim}));
      c->set_output(
          kPreservedVectorsIndex,
          c->MakeShape({input_size_dim, max_valency_dim, feature_width_dim}));

      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithSgdAndCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_embedding_table: float32")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(6));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithAdagradAndCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("accumulator: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(6));
      c->set_output(1, c->input(7));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("accumulator: float32")
    .Input("momenta: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Output("updated_momenta: float32")
    .Attr("use_nesterov: bool")
    .Attr("exponent: float")
    .Attr("beta1: float")
    .Attr("beta2: float")
    .Attr("epsilon: float")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(6));
      c->set_output(1, c->input(7));
      c->set_output(2, c->input(8));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithAdamAndCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("momenta: float32")
    .Input("velocity: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_embedding_table: float32")
    .Output("updated_momenta: float32")
    .Output("updated_velocity: float32")
    .Attr("use_sum_inside_sqrt: bool")
    .Attr("beta1: float")
    .Attr("beta2: float")
    .Attr("epsilon: float")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(6));
      c->set_output(1, c->input(7));
      c->set_output(2, c->input(8));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithFtrlAndCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("accumulator: float32")
    .Input("linear: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Output("updated_linear: float32")
    .Attr("multiply_linear_by_learning_rate: bool")
    .Attr("beta: float")
    .Attr("learning_rate_power: float")
    .Attr("l1_regularization_strength: float")
    .Attr("l2_regularization_strength: float")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(6));
      c->set_output(1, c->input(7));
      c->set_output(2, c->input(8));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseCoreSgd")
    .Input("indices: int32")
    .Input("gradient: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Output("updated_embedding_table: float32")
    .Attr("feature_width: int")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(3));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseCoreAdagrad")
    .Input("indices: int32")
    .Input("gradient: float32")
    .Input("learning_rate: float32")
    .Input("accumulator: float32")
    .Input("embedding_table: float32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Attr("feature_width: int")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(4));
      c->set_output(1, c->input(3));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseCoreAdagradMomentum")
    .Input("indices: int32")
    .Input("gradient: float32")
    .Input("learning_rate: float32")
    .Input("beta_1: float32")
    .Input("epsilon: float32")
    .Input("accumulator: float32")
    .Input("momentum: float32")
    .Input("embedding_table: float32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Output("updated_momentum: float32")
    .Attr("feature_width: int")
    .Attr("use_nesterov: bool")
    .Attr("beta_2: float")
    .Attr("exponent: float")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(7));
      c->set_output(1, c->input(5));
      c->set_output(2, c->input(6));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseCoreAdam")
    .Input("embedding_table: float32")
    .Input("indices: int32")
    .Input("gradient: float32")
    .Input("learning_rate: float32")
    .Input("momentum: float32")
    .Input("velocity: float32")
    .Input("beta_1: float32")
    .Input("beta_2: float32")
    .Input("epsilon: float32")
    .Output("updated_embedding_table: float32")
    .Output("updated_velocity: float32")
    .Output("updated_momentum: float32")
    // .Input("use_non_lazy_adam: bool")
    .Attr("feature_width: int")
    .Attr("use_sum_inside_sqrt: bool")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(5));
      c->set_output(2, c->input(4));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseCoreFtrl")
    .Input("embedding_table: float32")
    .Input("accumulator: float32")
    .Input("linear: float32")
    .Input("learning_rate: float32")
    .Input("indices: int32")
    .Input("gradient: float32")
    .Input("beta: float32")
    .Input("learning_rate_power: float32")
    .Input("l2_regularization_strength: float32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Output("updated_linear: float32")
    .Attr("feature_width: int")
    .Attr("multiply_linear_by_learning_rate: bool")
    .Attr("l1_regularization_strength: float")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return absl::OkStatus();
    });

REGISTER_OP("GlobalIterId")
    .Output("iter_id: int64")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulWithStaticBufferSize")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("embedding_table: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("activations: float32")
    .Attr("input_size: int >= 0")
    .Attr("quantization_config_low: float")
    .Attr("quantization_config_high: float")
    .Attr("quantization_config_num_buckets: int >= 0")
    .Attr("max_ids_per_sparse_core: int >= 1")
    .Attr("max_unique_ids_per_sparse_core: int >= 1")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int input_size;
      TF_RETURN_IF_ERROR(c->GetAttr("input_size", &input_size));
      shape_inference::ShapeHandle rank;
      for (int i = 0; i < 4; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &rank));
      }
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &rank));
      for (int i = 2; i < 4; ++i) {
        shape_inference::ShapeHandle merged;
        TF_RETURN_IF_ERROR(c->Merge(c->input(i), c->input(1), &merged));
      }
      shape_inference::ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(4), 0, c->MakeDim(input_size), &output_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &rank));
      c->set_output(0, output_shape);
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_embedding_table: float32")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("max_ids_per_sparse_core: int >= 1")
    .Attr("max_unique_ids_per_sparse_core: int >= 1")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(6));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("accumulator: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("max_ids_per_sparse_core: int >= 1")
    .Attr("max_unique_ids_per_sparse_core: int >= 1")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(6));
      c->set_output(1, c->input(7));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("accumulator: float32")
    .Input("momenta: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Output("updated_momenta: float32")
    .Attr("use_nesterov: bool")
    .Attr("exponent: float")
    .Attr("beta1: float")
    .Attr("beta2: float")
    .Attr("epsilon: float")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("max_ids_per_sparse_core: int >= 1")
    .Attr("max_unique_ids_per_sparse_core: int >= 1")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(6));
      c->set_output(1, c->input(7));
      c->set_output(2, c->input(8));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("momenta: float32")
    .Input("velocity: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_embedding_table: float32")
    .Output("updated_momenta: float32")
    .Output("updated_velocity: float32")
    .Attr("use_sum_inside_sqrt: bool")
    .Attr("beta1: float")
    .Attr("beta2: float")
    .Attr("epsilon: float")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("max_ids_per_sparse_core: int >= 1")
    .Attr("max_unique_ids_per_sparse_core: int >= 1")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(6));
      c->set_output(1, c->input(7));
      c->set_output(2, c->input(8));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("accumulator: float32")
    .Input("linear: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Output("updated_linear: float32")
    .Attr("multiply_linear_by_learning_rate: bool")
    .Attr("beta: float")
    .Attr("learning_rate_power: float")
    .Attr("l1_regularization_strength: float")
    .Attr("l2_regularization_strength: float")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("max_ids_per_sparse_core: int >= 1")
    .Attr("max_unique_ids_per_sparse_core: int >= 1")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      c->set_output(0, c->input(6));
      c->set_output(1, c->input(7));
      c->set_output(2, c->input(8));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulGradWithCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_gains: float32")
    .Input("activation_gradients: float32")
    .Input("tables: N * T")
    .Input("hyperparameters: M * float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("updated_tables: N * T")
    .Attr("N: int >= 1")
    .Attr("M: int >= 1")
    .Attr("custom_computation: func")
    .Attr("table_name: string")
    .Attr("num_sparsecores_per_device: int = -1")
    .Attr("T : {int32, float32} = DT_FLOAT")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int num_tables;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_tables));
      for (int i = 0; i < num_tables; ++i) {
        c->set_output(i, c->input(5));
      }
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulCustomCombinerOnTcGradWithSgdAndCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_pos_ids: int32")
    .Input("sorted_gains: float32")
    .Input("weights: float32")
    .Input("preserved_valencies: int32")
    .Input("preserved_vectors: float32")
    .Input("preserved_weights: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("combiner_weights_learning_rate: float32")
    .Input("embedding_table: float32")
    .Output("updated_embedding_table: float32")
    .Output("updated_weights: float32")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("max_valency: int >= 0")
    .Attr("num_weights: int >= 0")
    .Attr("combiner_table_vjp_computation: func")
    .Attr("combiner_weights_vjp_computation: func")
    .Attr("table_name: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      constexpr int kWeightsIndex = 5;
      constexpr int kPreservedValenciesIndex = 6;
      constexpr int kPreservedVectorsIndex = 7;
      constexpr int kPreservedWeightsIndex = 8;
      constexpr int kActivationGradientsIndex = 9;
      constexpr int kTablesIndex = 12;
      constexpr int kNumTables = 1;
      TF_RETURN_IF_ERROR(
          ValidateSparseDenseMatmulCustomCombinerGradWithCsrInputShape(
              c, kWeightsIndex, kPreservedValenciesIndex,
              kPreservedVectorsIndex, kPreservedWeightsIndex,
              kActivationGradientsIndex, kTablesIndex, kNumTables));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdagradAndCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_pos_ids: int32")
    .Input("sorted_gains: float32")
    .Input("weights: float32")
    .Input("preserved_valencies: int32")
    .Input("preserved_vectors: float32")
    .Input("preserved_weights: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("combiner_weights_learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("accumulator: float32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Output("updated_weights: float32")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("max_valency: int >= 0")
    .Attr("num_weights: int >= 0")
    .Attr("combiner_table_vjp_computation: func")
    .Attr("combiner_weights_vjp_computation: func")
    .Attr("table_name: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      constexpr int kWeightsIndex = 5;
      constexpr int kPreservedValenciesIndex = 6;
      constexpr int kPreservedVectorsIndex = 7;
      constexpr int kPreservedWeightsIndex = 8;
      constexpr int kActivationGradientsIndex = 9;
      constexpr int kTablesIndex = 12;
      constexpr int kNumTables = 2;
      TF_RETURN_IF_ERROR(
          ValidateSparseDenseMatmulCustomCombinerGradWithCsrInputShape(
              c, kWeightsIndex, kPreservedValenciesIndex,
              kPreservedVectorsIndex, kPreservedWeightsIndex,
              kActivationGradientsIndex, kTablesIndex, kNumTables));
      return absl::OkStatus();
    });

REGISTER_OP(
    "XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdagradMomentumAndCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_pos_ids: int32")
    .Input("sorted_gains: float32")
    .Input("weights: float32")
    .Input("preserved_valencies: int32")
    .Input("preserved_vectors: float32")
    .Input("preserved_weights: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("combiner_weights_learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("accumulator: float32")
    .Input("momenta: float32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Output("updated_momenta: float32")
    .Output("updated_weights: float32")
    .Attr("use_nesterov: bool")
    .Attr("exponent: float")
    .Attr("beta1: float")
    .Attr("beta2: float")
    .Attr("epsilon: float")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("max_valency: int >= 0")
    .Attr("num_weights: int >= 0")
    .Attr("combiner_table_vjp_computation: func")
    .Attr("combiner_weights_vjp_computation: func")
    .Attr("table_name: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      constexpr int kWeightsIndex = 5;
      constexpr int kPreservedValenciesIndex = 6;
      constexpr int kPreservedVectorsIndex = 7;
      constexpr int kPreservedWeightsIndex = 8;
      constexpr int kActivationGradientsIndex = 9;
      constexpr int kTablesIndex = 12;
      constexpr int kNumTables = 3;
      TF_RETURN_IF_ERROR(
          ValidateSparseDenseMatmulCustomCombinerGradWithCsrInputShape(
              c, kWeightsIndex, kPreservedValenciesIndex,
              kPreservedVectorsIndex, kPreservedWeightsIndex,
              kActivationGradientsIndex, kTablesIndex, kNumTables));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdamAndCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_pos_ids: int32")
    .Input("sorted_gains: float32")
    .Input("weights: float32")
    .Input("preserved_valencies: int32")
    .Input("preserved_vectors: float32")
    .Input("preserved_weights: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("combiner_weights_learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("momenta: float32")
    .Input("velocity: float32")
    .Output("updated_embedding_table: float32")
    .Output("updated_momenta: float32")
    .Output("updated_velocity: float32")
    .Output("updated_weights: float32")
    .Attr("use_sum_inside_sqrt: bool")
    .Attr("beta1: float")
    .Attr("beta2: float")
    .Attr("epsilon: float")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("max_valency: int >= 0")
    .Attr("num_weights: int >= 0")
    .Attr("combiner_table_vjp_computation: func")
    .Attr("combiner_weights_vjp_computation: func")
    .Attr("table_name: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      constexpr int kWeightsIndex = 5;
      constexpr int kPreservedValenciesIndex = 6;
      constexpr int kPreservedVectorsIndex = 7;
      constexpr int kPreservedWeightsIndex = 8;
      constexpr int kActivationGradientsIndex = 9;
      constexpr int kTablesIndex = 12;
      constexpr int kNumTables = 3;
      TF_RETURN_IF_ERROR(
          ValidateSparseDenseMatmulCustomCombinerGradWithCsrInputShape(
              c, kWeightsIndex, kPreservedValenciesIndex,
              kPreservedVectorsIndex, kPreservedWeightsIndex,
              kActivationGradientsIndex, kTablesIndex, kNumTables));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulCustomCombinerOnTcGradWithFtrlAndCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_pos_ids: int32")
    .Input("sorted_gains: float32")
    .Input("weights: float32")
    .Input("preserved_valencies: int32")
    .Input("preserved_vectors: float32")
    .Input("preserved_weights: float32")
    .Input("activation_gradients: float32")
    .Input("learning_rate: float32")
    .Input("combiner_weights_learning_rate: float32")
    .Input("embedding_table: float32")
    .Input("accumulator: float32")
    .Input("linear: float32")
    .Output("updated_embedding_table: float32")
    .Output("updated_accumulator: float32")
    .Output("updated_linear: float32")
    .Output("updated_weights: float32")
    .Attr("multiply_linear_by_learning_rate: bool")
    .Attr("beta: float")
    .Attr("learning_rate_power: float")
    .Attr("l1_regularization_strength: float")
    .Attr("l2_regularization_strength: float")
    .Attr("clip_weight_min: float = -inf")
    .Attr("clip_weight_max: float = inf")
    .Attr("max_valency: int >= 0")
    .Attr("num_weights: int >= 0")
    .Attr("combiner_table_vjp_computation: func")
    .Attr("combiner_weights_vjp_computation: func")
    .Attr("table_name: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      constexpr int kWeightsIndex = 5;
      constexpr int kPreservedValenciesIndex = 6;
      constexpr int kPreservedVectorsIndex = 7;
      constexpr int kPreservedWeightsIndex = 8;
      constexpr int kActivationGradientsIndex = 9;
      constexpr int kTablesIndex = 12;
      constexpr int kNumTables = 3;
      TF_RETURN_IF_ERROR(
          ValidateSparseDenseMatmulCustomCombinerGradWithCsrInputShape(
              c, kWeightsIndex, kPreservedValenciesIndex,
              kPreservedVectorsIndex, kPreservedWeightsIndex,
              kActivationGradientsIndex, kTablesIndex, kNumTables));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseDenseMatmulCustomCombinerOnTcGradWithCsrInput")
    .Input("row_pointers: int32")
    .Input("sorted_sample_ids: int32")
    .Input("sorted_token_ids: int32")
    .Input("sorted_pos_ids: int32")
    .Input("sorted_gains: float32")
    .Input("weights: float32")
    // We need to preserve the outputs of the SC forward pass and feed them into
    // the VJP computations in the backward pass.
    .Input("preserved_valencies: int32")
    .Input("preserved_vectors: float32")
    .Input("preserved_weights: float32")
    .Input("activation_gradients: float32")
    .Input("tables: N * float32")
    .Input("hyperparameters: M * float32")
    .Input("combiner_weights_learning_rate: float32")
    .Output("updated_tables: N * float32")
    .Output("updated_weights: float32")
    .Attr("N: int >= 1")
    .Attr("M: int >= 1")
    .Attr("max_valency: int >= 0")
    .Attr("num_weights: int >= 0")
    .Attr("combiner_table_vjp_computation: func")
    .Attr("combiner_weights_vjp_computation: func")
    .Attr("optimizer_custom_computation: func")
    .Attr("table_name: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      constexpr int kWeightsIndex = 5;
      constexpr int kPreservedValenciesIndex = 6;
      constexpr int kPreservedVectorsIndex = 7;
      constexpr int kPreservedWeightsIndex = 8;
      constexpr int kActivationGradientsIndex = 9;
      constexpr int kTablesIndex = 10;
      int num_tables;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_tables));
      TF_RETURN_IF_ERROR(
          ValidateSparseDenseMatmulCustomCombinerGradWithCsrInputShape(
              c, kWeightsIndex, kPreservedValenciesIndex,
              kPreservedVectorsIndex, kPreservedWeightsIndex,
              kActivationGradientsIndex, kTablesIndex, num_tables));
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseActivationsUnstack")
    .Attr("num_tables: int >= 1")
    .Attr("sample_counts: list(int) >= 1")
    .Attr("features: list(int) >= 1")
    .Attr("interleaved: bool")
    .Attr("input_dtype: type")
    .Attr("dtype: type")
    .Input("stacked_activations: input_dtype")
    .Output("unstacked_activations: num_tables * dtype")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int num_tables;
      TF_RETURN_IF_ERROR(c->GetAttr("num_tables", &num_tables));
      std::vector<int> sample_counts;
      TF_RETURN_IF_ERROR(c->GetAttr("sample_counts", &sample_counts));
      if (sample_counts.size() != num_tables) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid number of sample counts. Expected: %d, got: %d",
            num_tables, sample_counts.size()));
      }
      std::vector<int> features;
      TF_RETURN_IF_ERROR(c->GetAttr("features", &features));
      if (features.size() != num_tables) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid number of features. Expected: %d, got: %d",
                            num_tables, features.size()));
      }
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("input_dtype", &dtype));
      if (dtype != DT_FLOAT) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported dtype for stacked activations: %s",
                            DataType_Name(dtype)));
      }
      for (int i = 0; i < num_tables; ++i) {
        shape_inference::ShapeHandle unstacked_activation_shape =
            c->Matrix(sample_counts[i], features[i]);
        c->set_output(i, unstacked_activation_shape);
      }
      return absl::OkStatus();
    });

REGISTER_OP("XlaSparseGradientsStack")
    .Attr("num_tables: int >= 1")
    .Attr("interleaved: bool")
    .Attr("input_dtype: type")
    .Attr("dtype: type")
    .Input("unstacked_gradients: num_tables * input_dtype")
    .Output("stacked_gradients: dtype")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int num_tables;
      TF_RETURN_IF_ERROR(c->GetAttr("num_tables", &num_tables));
      if (c->num_inputs() != num_tables) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid number of gradients. Expected: %d, got: %d", num_tables,
            c->num_inputs()));
      }
      int total_sample_count = 0;
      std::vector<int> features(num_tables);
      for (int i = 0; i < num_tables; ++i) {
        features[i] = c->Value(c->Dim(c->input(i), 1));
        total_sample_count += c->Value(c->Dim(c->input(0), 0));
      }
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &dtype));
      if (dtype != DT_FLOAT) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported dtype for stacked gradients: %s",
                            DataType_Name(dtype)));
      }
      int padded_feature = xla::RoundUpTo(*absl::c_max_element(features), 8);
      shape_inference::ShapeHandle stacked_gradients_shape =
          c->Matrix(total_sample_count, padded_feature);
      c->set_output(0, stacked_gradients_shape);
      return absl::OkStatus();
    });

}  // namespace tensorflow
