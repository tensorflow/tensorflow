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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .Input("embedding_table: float32")
    .Input("num_minibatches_per_physical_sparse_core: int32")
    .Output("activations: float32")
    .Attr("input_size: int >= 0")
    .Attr("quantization_config_low: float")
    .Attr("quantization_config_high: float")
    .Attr("quantization_config_num_buckets: int >= 0")
    .Attr("table_name: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
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
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return absl::OkStatus();
    });

REGISTER_OP("GlobalIterId")
    .Output("iter_id: int64")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

}  // namespace tensorflow
