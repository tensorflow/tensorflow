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

#include "tensorflow/contrib/tpu/proto/tpu_embedding_config.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

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
// 2. Use the TPUEmbeddingLoad Op to initialize the embedding tables in TPU
//    memories, sharded across the memories attached to each Host.
// 3. Use TPUEmbeddingEnqueueSparseBatch to provide the TPU with embedding
//    indices and aggregation weights.
// 4. TPUEmbeddingReceiveActivations returns a list of Tensors, containing the
//    activations from each table specified in the configuration.
// 5. TPUEmbeddingActivations, when used with appropriate Python libraries,
//    enables the automatic differentiation of models that use embeddings.
// 6. TPUEmbeddingSendGradients takes a list of Tensors (of the same shapes
//    as those returned by TPUEmbeddingReceivActivations) containing gradients
//    to use in updating the embedding tables.
// 7. Before saving a checkpoint, use the TPUEmbeddingRetrieve Op to update
//    the Graph's embedding table Variables from the updated tables in the
//    TPU memories.
//
// TPU Embeddings use dedicated ops to enforce Host/TPU consistency in the
// state of embedding table variables. Before beginning training or inference,
// the model must Load the optimizer parameters into the TPU memories. Before
// saving a checkpoint, the model must Retreieve the parameters back into the
// host CPU memory.

REGISTER_OP("TPUEmbeddingLoadGradientDescentParameters")
    .Input("parameters: float32")
    .Attr("tpu_embedding_config: string")
    .Attr("table_id: int >= 0")
    .Attr("num_hosts: int >= 1")
    .Attr("host_id: int >= 0")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Load an embedding table shard into TPU memory for use with GradientDescent.

TPU embeddings use dedicated per-optimizer Ops for loading and retrieving 
trainable variables and optimizer state from TPU memory. This op enables
functionality equivalent to GradientDescentOptimizer.

parameters: The shard of the embedding table resident on the host executing this
    op. For single-TPU models, this is the entire embedding table.
tpu_embedding_config: Serialized TPUEmbeddingConfiguration proto.
table_id: The id of the table specified in the tpu_embedding_config.
num_hosts: The number of CPU hosts in the distributed training job.
host_id: Which CPU host in the distributed training job will execute this op.
)doc");

namespace tpu_embedding_config_util {

Status GradientDescentShapes(shape_inference::InferenceContext *c) {
  string config_string;
  TF_RETURN_IF_ERROR(c->GetAttr("tpu_embedding_config", &config_string));
  tpu::TPUEmbeddingConfiguration config;
  if (!config.ParseFromString(config_string)) {
    return errors::InvalidArgument("Malformed tpu_embedding_config.");
  }

  int table_id;
  TF_RETURN_IF_ERROR(c->GetAttr("table_id", &table_id));
  int64 num_tables = config.table_config_size();
  if (table_id >= num_tables) {
    return errors::InvalidArgument("Table id >= num_tables");
  }
  int64 width = config.table_config(table_id).width();
  int64 num_rows = config.table_config(table_id).num_rows();

  TF_RETURN_IF_ERROR(c->set_output("parameters", {c->Matrix(num_rows, width)}));
  return Status::OK();
}

}  // namespace tpu_embedding_config_util

REGISTER_OP("TPUEmbeddingRetrieveGradientDescentParameters")
    .Output("parameters: float32")
    .Attr("tpu_embedding_config: string")
    .Attr("table_id: int")
    .Attr("num_hosts: int")
    .Attr("host_id: int")
    .SetIsStateful()
    .SetShapeFn(tpu_embedding_config_util::GradientDescentShapes)
    .Doc(R"doc(
Retrieve an embedding table shard from TPU memory.

TPU embeddings use dedicated per-optimizer Ops for loading and retrieving 
trainable variables and optimizer state from TPU memory. This op enables
functionality equivalent to GradientDescentOptimizer.

tpu_embedding_config: Serialized TPUEmbeddingConfiguration proto.
table_id: The id of the table specified in tpu_embedding_config.
num_hosts: The number of CPU hosts in the distributed training job.
host_id: Which CPU host in the distributed training job will execute this op.
)doc");

REGISTER_OP("TPUEmbeddingLoadAdagradParameters")
    .Input("parameters: float32")
    .Input("accumulators: float32")
    .Attr("tpu_embedding_config: string")
    .Attr("table_id: int >= 0")
    .Attr("num_hosts: int >= 1")
    .Attr("host_id: int >= 0")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Load an embedding table shard into TensorNode memories for use with Adagrad.

TPU embeddings use dedicated per-optimizer Ops for loading and retrieving
trainable variables and optimizer state from TPU memory. This op enables
functionality equivalent to AdagradOptimizer.

parameters: The shard of the embedding table resident on the host executing this
    op. For single-TPU models, this is the entire embedding table.
accumulators: Shard of the Adagrad accumulators resident on the host executing
    this op.
tpu_embedding_config: Serialized TPUEmbeddingConfiguration proto.
table_id: The id of the table specified in the embedding_config.
num_hosts: The number of CPU hosts in the distributed training job.
host_id: Which CPU host in the distributed training job will execute this op.
)doc");

namespace tpu_embedding_config_util {

Status AdagradShapes(shape_inference::InferenceContext *c) {
  string config_string;
  TF_RETURN_IF_ERROR(c->GetAttr("tpu_embedding_config", &config_string));
  tpu::TPUEmbeddingConfiguration config;
  if (!config.ParseFromString(config_string)) {
    return errors::InvalidArgument("Malformed tpu_embedding_config.");
  }

  int table_id;
  TF_RETURN_IF_ERROR(c->GetAttr("table_id", &table_id));
  int64 num_tables = config.table_config_size();
  if (table_id >= num_tables) {
    return errors::InvalidArgument("Table id >= num_tables");
  }
  int64 width = config.table_config(table_id).width();
  int64 num_rows = config.table_config(table_id).num_rows();

  TF_RETURN_IF_ERROR(c->set_output("parameters", {c->Matrix(num_rows, width)}));
  TF_RETURN_IF_ERROR(
      c->set_output("accumulators", {c->Matrix(num_rows, width)}));
  return Status::OK();
}

}  // namespace tpu_embedding_config_util

REGISTER_OP("TPUEmbeddingRetrieveAdagradParameters")
    .Output("parameters: float32")
    .Output("accumulators: float32")
    .Attr("tpu_embedding_config: string")
    .Attr("table_id: int >= 0")
    .Attr("num_hosts: int >= 1")
    .Attr("host_id: int >= 0")
    .SetIsStateful()
    .SetShapeFn(tpu_embedding_config_util::AdagradShapes)
    .Doc(R"doc(
Retrieve an embedding table shard from TPU memory.

TPU embeddings use dedicated per-optimizer Ops for loading and retrieving 
trainable variables and optimizer state from TPU memory. This op enables
functionality equivalent to AdagradOptimizer.

tpu_embedding_config: Serialized TPUEmbeddingConfiguration proto.
table_id: The id of the table specified in the embedding_config_json.
num_hosts: The number of CPU hosts in the distributed training job.
host_id: Which CPU host in the distributed training job will execute this op.
)doc");

REGISTER_OP("TPUEmbeddingEnqueueSparseBatch")
    .Input("sample_indices: num_tables * int32")
    .Input("embedding_indices: num_tables * int32")
    .Input("aggregation_weights: num_tables * float32")
    .Attr("num_tables: int")
    .Attr("device_ordinal: int = -1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
An op that feeds a batch of embedding indices and weights to the TPU.

Embedding lookups are equivalent to sparse-dense matrix multiplications: the
sparse matrix contains nonzeros in column j in order to retrieve row j from the
embedding table.

The three Tensor list arguments (sample_indices, embedding_indices, and
aggregation_weights) represent these sparse matrices in COO format. The Tensor
lists each have one entry for each embedding table specified in the model.
For the kth embedding table, the three Tensors at position k in the list
specify a COO-format sparse matrix. For the kth table, the row indices,
column indices, and nonzero values of the COO sparse matrix are specified by
sample_indices[k], embedding_indices[k], and aggregation_weights[k],
respectively. Entries must be sorted by row index, then by column index.

There should be at most one TPUEmbeddingEnqueueSparseBatch op in a signle
training step per TPU shard.

sample_indices: A list of rank 1 Tensors specifying row indices of the COO
    sparse matrix representing the embedding lookups for each table.
embedding_indices: A list of rank 1 Tensors  specifying column indices of the
    COO sparse matrix representing the embedding lookups for each table.
aggregation_weights: A list of rank 1 Tensors specifying the nonzero values
    of the COO sparse matrix representing the embedding lookups for each table.
device_ordinal: The TPU device to use. This should be -1 when the Op
    is running on a TPU device, and >= 0 when the Op is running on the CPU
    device.
)doc");

namespace tpu_embedding_config_util {

Status ActivationShapes(shape_inference::InferenceContext *c) {
  string config_string;
  TF_RETURN_IF_ERROR(c->GetAttr("tpu_embedding_config", &config_string));
  tpu::TPUEmbeddingConfiguration config;
  if (!config.ParseFromString(config_string)) {
    return errors::InvalidArgument("Malformed tpu_embedding_config.");
  }
  int64 batch_size = config.batch_size();
  int64 num_tables = config.table_config_size();
  for (int table_id = 0; table_id < num_tables; ++table_id) {
    int64 width = config.table_config(table_id).width();
    int64 num_features = config.table_config(table_id).num_features();
    c->set_output(table_id, c->Matrix(batch_size * num_features, width));
  }
  return Status::OK();
}

}  // namespace tpu_embedding_config_util

REGISTER_OP("TPUEmbeddingReceiveActivations")
    .Output("outputs: num_tables * float")
    .Attr("num_tables: int >= 1")
    .Attr("tpu_embedding_config: string")
    .SetIsStateful()
    .SetShapeFn(tpu_embedding_config_util::ActivationShapes)
    .Doc(R"doc(
An op that receives embeddng activations on the TPU.

The TPU system performs the embedding lookups and aggregations specified by
the arguments to TPUEmbeddingEnqueueSparseBatch. The results of these
aggregations are visible to the Tensorflow Graph as the outputs of a
TPUEmbeddingDequeueActivations Op. This op returns a list containing one
Tensor of activations per table specified in the model. There can be at most
one ReceieveActivations op in the TPU graph.

outputs: A TensorList of embedding activations containing one Tensor per
    embedding table in the model.
num_tables: The number of output activation tensors, equal to the number of
    embedding tables in the model.
tpu_embedding_config: Serialized TPUEmbeddingConfiguration proto.
)doc");

REGISTER_OP("TPUEmbeddingActivations")
    .Input("embedding_variable: float32")
    .Input("sliced_activations: float32")
    .Output("output: float32")
    .Attr("table_id: int >= 0")
    .Attr("lookup_id: int >= 0")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
An op enabling differentiation of TPU Embeddings.

This op simply returns its first input, which is assumed to have been sliced
from the Tensors returnd by TPUEmbeddingDequeueActivations. The presence of this
op, and its first argument being a trainable Variable, enables automatic
differentiation of graphs containing embeddings via the TPU Embedding Python
libraries.

embedding_variable: A trainable variable, enabling optimizers to find this op.
sliced_activations: The embedding activations Tensor to return.
table_id: The id of the table in the embedding layer configuration from which
    these activations were computed.
lookup_id: Identifier of the set of embedding indices which produced these
    activations.
)doc");

REGISTER_OP("TPUEmbeddingSendGradients")
    .Input("gradients: num_tables * float32")
    .Attr("num_tables: int >= 1")
    .Attr("tpu_embedding_config: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
An op that performs gradient updates of embedding tables.

The TensorList argument has the same length and shapes as the return value of
TPUEmbeddingReceiveActivations, but contains gradients of the model's loss
with respect to the embedding activations. The embedding tables are updated
from these gradients via the optimizer specified in the configuration given
to tpu.initialize_system.

gradients: A TensorList of gradients with which to update embedding tables.
tpu_embedding_config: Serialized TPUEmbeddingConfiguration proto.
)doc");

}  // namespace tensorflow
