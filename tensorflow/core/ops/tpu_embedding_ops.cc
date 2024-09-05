/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/tpu_embedding_output_layout_utils.h"

namespace tensorflow {

// TPUs use a specialized mechanism for performing embedding lookups,
// necessitating differences in TF Graphs that use embeddings on TPUs relative
// to CPUs. Embedding lookups on TPU systems are achieved by including the
// following in the TF Graph.
//
// 0. Construct a TPUEmbeddingConfiguration, specifying the embedding tables
//    in the model, the size of the TPU system to be used, and the optimizer to
//    be used for each table. Some of this information is redundant with other
//    pieces of the TF Graph.
// 1. Pass this TPUEmbeddingConfiguration to tpu.initialize_system() as the
//    tpu_embedding_config parameter.
// 2. Use the LoadTPUEmbedding Ops to initialize the embedding tables in TPU
//    memories, sharded across the memories attached to each Host.
// 3. Use EnqueueTPUEmbeddingSparseBatch to provide the TPU with embedding
//    indices and aggregation weights.
// 4. RecvTPUEmbeddingActivations returns a list of Tensors, containing the
//    activations from each table specified in the configuration.
// 5. TPUEmbeddingActivations, when used with appropriate Python libraries,
//    enables the automatic differentiation of models that use embeddings.
// 6. SendTPUEmbeddingGradients takes a list of Tensors (of the same shapes
//    as those returned by TPUEmbeddingReceiveActivations) containing gradients
//    to use in updating the embedding tables.
// 7. Before saving a checkpoint, use the RetrieveTPUEmbedding Ops to update
//    the Graph's embedding table Variables from the updated tables in the
//    TPU memories.
//
// TPU Embeddings use dedicated ops to enforce Host/TPU consistency in the
// state of embedding table variables. Before beginning training or inference,
// the model must Load the optimizer parameters into the TPU memories. Before
// saving a checkpoint, the model must Retrieve the parameters back into the
// host CPU memory.

REGISTER_OP("RecvTPUEmbeddingActivations")
    .Output("outputs: num_outputs * float32")
    .Attr("num_outputs: int >= 1")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
      std::string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      tpu::TPUEmbeddingConfiguration config;
      if (!config.ParseFromString(config_string)) {
        return errors::InvalidArgument("Malformed tpu_embedding_config.");
      }
      std::vector<TensorShapeProto> output_shapes;
      TF_RETURN_IF_ERROR(ComputeOutputTensorShapes(config, &output_shapes));
      if (c->num_outputs() != output_shapes.size()) {
        return errors::InvalidArgument("num outputs != size of output shapes");
      }
      for (int i = 0; i < c->num_outputs(); ++i) {
        shape_inference::ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromShapeProto(output_shapes[i], &output_shape));
        c->set_output(i, output_shape);
      }
      return absl::OkStatus();
    });

REGISTER_OP("TPUEmbeddingActivations")
    .Input("embedding_variable: float32")
    .Input("sliced_activations: float32")
    .Output("output: float32")
    .Attr("table_id: int >= 0")
    .Attr("lookup_id: int >= 0")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(1));
      return absl::OkStatus();
    });

REGISTER_OP("SendTPUEmbeddingGradients")
    .Input("inputs: N * float32")
    .Input("learning_rates: NN * float32")
    .Attr("N: int >= 1")
    .Attr("NN: int >= 0 = 0")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
      int nn;
      TF_RETURN_IF_ERROR(c->GetAttr("NN", &nn));
      std::vector<shape_inference::ShapeHandle> learning_rates;
      TF_RETURN_IF_ERROR(c->input("learning_rates", &learning_rates));
      for (int i = 0; i < nn; ++i) {
        // Verify that each learning_rates element is scalar
        shape_inference::ShapeHandle learning_rates_shape;
        TF_RETURN_IF_ERROR(
            c->WithRank(learning_rates[i], 0, &learning_rates_shape));
      }

      return absl::OkStatus();
    });

REGISTER_OP("EnqueueTPUEmbeddingIntegerBatch")
    .Input("batch: N * int32")
    .Input("mode_override: string")
    .Attr("N: int >= 1")
    .Attr("device_ordinal: int = -1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("EnqueueTPUEmbeddingSparseBatch")
    .Input("sample_indices: N * T1")
    .Input("embedding_indices: N * T2")
    .Input("aggregation_weights: N * T3")
    .Input("mode_override: string")
    .Attr("T1: {int32,int64} = DT_INT32")
    .Attr("T2: {int32,int64} = DT_INT32")
    .Attr("T3: {float32,float64} = DT_FLOAT")
    .Attr("N: int >= 1")
    .Attr("device_ordinal: int = -1")
    .Attr("combiners: list(string) = []")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
      std::vector<string> combiners;
      TF_RETURN_IF_ERROR(c->GetAttr("combiners", &combiners));
      int n;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
      if (!combiners.empty() && combiners.size() != n) {
        return errors::InvalidArgument("Invalid length of combiners. Have ",
                                       combiners.size(), " but expected 0 or ",
                                       n);
      }

      return absl::OkStatus();
    });

REGISTER_OP("EnqueueTPUEmbeddingSparseTensorBatch")
    .Input("sample_indices: N * T1")
    .Input("embedding_indices: N * T2")
    .Input("aggregation_weights: N * T3")
    .Input("mode_override: string")
    .Attr("T1: {int32,int64} = DT_INT32")
    .Attr("T2: {int32,int64} = DT_INT32")
    .Attr("T3: {float32,float64} = DT_FLOAT")
    .Attr("N: int >= 1")
    .Attr("device_ordinal: int = -1")
    .Attr("combiners: list(string) = []")
    .Attr("table_ids: list(int)")
    .Attr("max_sequence_lengths: list(int) = []")
    .Attr("num_features: list(int) = []")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("EnqueueTPUEmbeddingRaggedTensorBatch")
    .Input("sample_splits: N * T1")
    .Input("embedding_indices: N * T2")
    .Input("aggregation_weights: N * T3")
    .Input("mode_override: string")
    .Attr("T1: {int32,int64} = DT_INT32")
    .Attr("T2: {int32,int64} = DT_INT32")
    .Attr("T3: {float32,float64} = DT_FLOAT")
    .Attr("N: int >= 1")
    .Attr("device_ordinal: int = -1")
    .Attr("combiners: list(string) = []")
    .Attr("table_ids: list(int)")
    .Attr("max_sequence_lengths: list(int) = []")
    .Attr("num_features: list(int) = []")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("DynamicEnqueueTPUEmbeddingRaggedTensorBatch")
    .Input("sample_splits: N * T1")
    .Input("embedding_indices: N * T2")
    .Input("aggregation_weights: N * T3")
    .Input("mode_override: string")
    .Input("device_ordinal: int32")
    .Attr("T1: {int32,int64} = DT_INT32")
    .Attr("T2: {int32,int64} = DT_INT32")
    .Attr("T3: {float32,float64} = DT_FLOAT")
    .Attr("N: int >= 1")
    .Attr("combiners: list(string) = []")
    .Attr("table_ids: list(int)")
    .Attr("max_sequence_lengths: list(int) = []")
    .Attr("num_features: list(int) = []")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("EnqueueTPUEmbeddingArbitraryTensorBatch")
    .Input("sample_indices_or_row_splits: N * T1")
    .Input("embedding_indices: N * T2")
    .Input("aggregation_weights: N * T3")
    .Input("mode_override: string")
    .Attr("T1: {int32,int64} = DT_INT32")
    .Attr("T2: {int32,int64} = DT_INT32")
    .Attr("T3: {float32,float64} = DT_FLOAT")
    .Attr("N: int >= 1")
    .Attr("device_ordinal: int = -1")
    .Attr("combiners: list(string) = []")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")
    .Input("sample_indices_or_row_splits: N * T1")
    .Input("embedding_indices: N * T2")
    .Input("aggregation_weights: N * T3")
    .Input("mode_override: string")
    .Input("device_ordinal: int32")
    .Attr("T1: {int32,int64} = DT_INT32")
    .Attr("T2: {int32,int64} = DT_INT32")
    .Attr("T3: {float32,float64} = DT_FLOAT")
    .Attr("N: int >= 1")
    .Attr("combiners: list(string) = []")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow
