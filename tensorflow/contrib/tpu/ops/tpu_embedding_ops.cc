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

#include "tensorflow/contrib/tpu/proto/tpu_embedding_configuration.pb.h"
#include "tensorflow/contrib/tpu/utils/tpu_embedding_optimization_parameters_utils.h"
#include "tensorflow/contrib/tpu/utils/tpu_embedding_output_layout_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

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
//    as those returned by TPUEmbeddingReceiveActivations) containing gradients
//    to use in updating the embedding tables.
// 7. Before saving a checkpoint, use the TPUEmbeddingRetrieve Op to update
//    the Graph's embedding table Variables from the updated tables in the
//    TPU memories.
//
// TPU Embeddings use dedicated ops to enforce Host/TPU consistency in the
// state of embedding table variables. Before beginning training or inference,
// the model must Load the optimizer parameters into the TPU memories. Before
// saving a checkpoint, the model must Retrieve the parameters back into the
// host CPU memory.

namespace {

void RegisterPerTableLoadAndRetrieveOps();

class RegisterPerTableLoadAndRetrieveOpsOnConstruction {
 public:
  RegisterPerTableLoadAndRetrieveOpsOnConstruction() {
    RegisterPerTableLoadAndRetrieveOps();
  }
};

// Object whose constructor does registrations.
RegisterPerTableLoadAndRetrieveOpsOnConstruction
    register_per_table_load_and_retrieve_ops_var;

Status RegisterPerTableLoadOpsForAlgorithmBody(
    tpu::OptimizationAlgorithm alg, bool is_debug_op,
    OpRegistrationData* op_reg_data) {
  tpu::GradientAccumulationSupport grad_accum_support;
  TF_CHECK_OK(GetGradientAccumulationSupport(alg, &grad_accum_support));

  std::vector<tpu::StateVariableSpecification> state_variable_specs;
  TF_CHECK_OK(GetOptimizationAlgorithmStateVariables(
      alg,
      grad_accum_support == tpu::GradientAccumulationSupport::kSupported &&
          is_debug_op,
      &state_variable_specs));
  auto* op_def = &op_reg_data->op_def;
  op_def->set_name(
      strings::StrCat("LoadTPUEmbedding", GetOptimizationAlgorithmName(alg),
                      "Parameters", (is_debug_op ? "GradAccumDebug" : "")));
  // It is important for the order of the inputs to the op defined here
  // to match the order in input_names because the indexes are used in
  // the combining transformation.
  for (const auto& parameter : state_variable_specs) {
    if (parameter.has_user_defined() || is_debug_op) {
      auto* arg = op_def->add_input_arg();
      arg->set_name(parameter.name());
      arg->set_description(
          strings::StrCat("Value of ", parameter.name(), " used in the ",
                          GetOptimizationAlgorithmFriendlyName(alg),
                          " optimization algorithm."));
      arg->set_type(DT_FLOAT);
    }
  }
  {
    auto* table_id_attr = op_def->add_attr();
    table_id_attr->set_name("table_id");
    table_id_attr->set_type("int");
    table_id_attr->set_has_minimum(true);
    table_id_attr->set_minimum(-1);
    table_id_attr->mutable_default_value()->set_i(-1);
  }
  {
    auto* table_name_attr = op_def->add_attr();
    table_name_attr->set_name("table_name");
    table_name_attr->set_type("string");
    table_name_attr->mutable_default_value()->set_s("");
  }
  {
    auto* num_shards_attr = op_def->add_attr();
    num_shards_attr->set_name("num_shards");
    num_shards_attr->set_type("int");
  }
  {
    auto* shard_id_attr = op_def->add_attr();
    shard_id_attr->set_name("shard_id");
    shard_id_attr->set_type("int");
  }
  op_def->set_summary("Load embedding parameters for a single table.");
  string parameter_descriptions;
  for (const auto& parameter : state_variable_specs) {
    if (parameter.has_user_defined() || is_debug_op) {
      strings::Appendf(&parameter_descriptions,
                       R"(
%s: A tensor containing the initial embedding table %s to use in embedding
lookups using the %s optimization algorithm.)",
                       parameter.name().c_str(), parameter.name().c_str(),
                       GetOptimizationAlgorithmFriendlyName(alg).c_str());
    }
  }
  op_def->set_description(strings::Printf(R"doc(
An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.
%s
table_name: Name of this table; must match a name in the
  TPUEmbeddingConfiguration proto (overrides table_id).
num_shards: Number of shards into which the embedding tables are divided.
shard_id: Identifier of shard for this operation.
table_id: Index of this table in the EmbeddingLayerConfiguration proto
  (deprecated).
)doc",
                                          parameter_descriptions.c_str()));
  op_def->set_is_commutative(false);
  op_def->set_is_aggregate(false);
  op_def->set_is_stateful(true);
  auto shape_inference_function =
      [state_variable_specs,
       is_debug_op](shape_inference::InferenceContext* c) -> Status {
    int table_id;
    TF_RETURN_IF_ERROR(c->GetAttr("table_id", &table_id));
    string table_name;
    TF_RETURN_IF_ERROR(c->GetAttr("table_name", &table_name));
    // Exactly one must be non-default.
    if ((table_id >= 0) == (!table_name.empty())) {
      return errors::InvalidArgument(
          "exactly one of table_id or table_name must be non-default");
    }
    int num_shards;
    TF_RETURN_IF_ERROR(c->GetAttr("num_shards", &num_shards));
    int shard_id;
    TF_RETURN_IF_ERROR(c->GetAttr("shard_id", &shard_id));
    const int user_param_count =
        std::count_if(state_variable_specs.begin(), state_variable_specs.end(),
                      [&](const tpu::StateVariableSpecification& sv) {
                        return sv.has_user_defined() || is_debug_op;
                      });
    std::vector<shape_inference::ShapeHandle> inputs(user_param_count);
    int input_index = 0;
    for (int i = 0; i < state_variable_specs.size(); ++i) {
      if (state_variable_specs[i].has_user_defined() || is_debug_op) {
        std::vector<shape_inference::ShapeHandle> input_temp;
        TF_RETURN_IF_ERROR(
            c->input(state_variable_specs[i].name(), &input_temp));
        if (input_temp.size() != 1) {
          return errors::InvalidArgument("each input to be rank 1");
        }
        inputs[input_index] = input_temp[0];
        ++input_index;
      }
    }
    // Verify shapes have rank 2 and are compatible when they are
    // required to be valid.
    shape_inference::ShapeHandle parameter_shape;
    TF_RETURN_IF_ERROR(c->WithRank(inputs[0], 2, &parameter_shape));
    for (int j = 1; j < user_param_count; ++j) {
      shape_inference::ShapeHandle accumulator_j_shape;
      TF_RETURN_IF_ERROR(c->WithRank(inputs[j], 2, &accumulator_j_shape));
      shape_inference::ShapeHandle merged;
      TF_RETURN_IF_ERROR(
          c->Merge(parameter_shape, accumulator_j_shape, &merged));
    }
    return Status::OK();
  };
  op_reg_data->shape_inference_fn = shape_inference_function;
  return Status::OK();
}

Status RegisterPerTableRetrieveOpsForAlgorithmBody(
    tpu::OptimizationAlgorithm alg, bool is_debug_op,
    OpRegistrationData* op_reg_data) {
  tpu::GradientAccumulationSupport grad_accum_support;
  TF_CHECK_OK(GetGradientAccumulationSupport(alg, &grad_accum_support));

  std::vector<tpu::StateVariableSpecification> state_variable_specs;
  TF_CHECK_OK(GetOptimizationAlgorithmStateVariables(
      alg,
      grad_accum_support == tpu::GradientAccumulationSupport::kSupported &&
          is_debug_op,
      &state_variable_specs));

  auto* op_def = &op_reg_data->op_def;
  op_def->set_name(strings::StrCat(
      "RetrieveTPUEmbedding", tpu::GetOptimizationAlgorithmName(alg),
      "Parameters", (is_debug_op ? "GradAccumDebug" : "")));
  // It is important for the order of the outputs of the op defined here
  // to match the order in output_names because the indexes are used in
  // the combining transformation.
  for (const auto& parameter : state_variable_specs) {
    if (parameter.has_user_defined() || is_debug_op) {
      auto* arg = op_def->add_output_arg();
      arg->set_name(parameter.name());
      arg->set_description(
          strings::StrCat("Parameter ", parameter.name(), " updated by the ",
                          tpu::GetOptimizationAlgorithmFriendlyName(alg),
                          " optimization algorithm."));
      arg->set_type(DT_FLOAT);
    }
  }
  {
    auto* table_id_attr = op_def->add_attr();
    table_id_attr->set_name("table_id");
    table_id_attr->set_type("int");
    table_id_attr->set_has_minimum(true);
    table_id_attr->set_minimum(-1);
    table_id_attr->mutable_default_value()->set_i(-1);
  }
  {
    auto* table_name_attr = op_def->add_attr();
    table_name_attr->set_name("table_name");
    table_name_attr->set_type("string");
    table_name_attr->mutable_default_value()->set_s("");
  }
  {
    auto* num_shards_attr = op_def->add_attr();
    num_shards_attr->set_name("num_shards");
    num_shards_attr->set_type("int");
  }
  {
    auto* shard_id_attr = op_def->add_attr();
    shard_id_attr->set_name("shard_id");
    shard_id_attr->set_type("int");
  }
  op_def->set_summary("Retrieve embedding parameters for a single table.");
  string parameter_descriptions;
  for (const auto& param : state_variable_specs) {
    if (param.has_user_defined() || is_debug_op) {
      strings::Appendf(&parameter_descriptions,
                       R"(
%s: A tensor containing the embedding table %s to store with the
parameters from embedding updates using the %s optimization algorithm.)",
                       param.name().c_str(), param.name().c_str(),
                       tpu::GetOptimizationAlgorithmFriendlyName(alg).c_str());
    }
  }
  op_def->set_description(strings::Printf(R"doc(
An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.
%s
table_name: Name of this table; must match a name in the
  TPUEmbeddingConfiguration proto (overrides table_id).
num_shards: Number of shards into which the embedding tables are divided.
shard_id: Identifier of shard for this operation.
table_id: Index of this table in the EmbeddingLayerConfiguration proto
  (deprecated).
)doc",
                                          parameter_descriptions.c_str()));
  op_def->set_is_commutative(false);
  op_def->set_is_aggregate(false);
  op_def->set_is_stateful(true);
  auto shape_inference_function =
      [state_variable_specs,
       is_debug_op](shape_inference::InferenceContext* c) -> Status {
    int table_id;
    TF_RETURN_IF_ERROR(c->GetAttr("table_id", &table_id));
    string table_name;
    TF_RETURN_IF_ERROR(c->GetAttr("table_name", &table_name));
    // Exactly one must be non-default.
    if ((table_id >= 0) == (!table_name.empty())) {
      return errors::InvalidArgument(
          "exactly one of table_id or table_name must be non-default");
    }
    int num_shards;
    TF_RETURN_IF_ERROR(c->GetAttr("num_shards", &num_shards));
    int shard_id;
    TF_RETURN_IF_ERROR(c->GetAttr("shard_id", &shard_id));
    for (int j = 0; j < state_variable_specs.size(); ++j) {
      if (state_variable_specs[j].has_user_defined() || is_debug_op) {
        auto shape = c->MakeShape(
            std::vector<shape_inference::DimensionHandle>(2, c->UnknownDim()));
        TF_RETURN_IF_ERROR(
            c->set_output(state_variable_specs[j].name(),
                          std::vector<shape_inference::ShapeHandle>(1, shape)));
      }
    }
    return Status::OK();
  };
  op_reg_data->shape_inference_fn = shape_inference_function;
  return Status::OK();
}

void RegisterPerTableLoadAndRetrieveOps() {
  // Load ops
  for (tpu::OptimizationAlgorithm alg : tpu::GetOptimizationAlgorithms()) {
    OpRegistry::Global()->Register(
        [alg](OpRegistrationData* op_reg_data) -> Status {
          return RegisterPerTableLoadOpsForAlgorithmBody(alg, false,
                                                         op_reg_data);
        });
    tpu::GradientAccumulationSupport grad_accum_support;
    TF_CHECK_OK(GetGradientAccumulationSupport(alg, &grad_accum_support));
    if (grad_accum_support == tpu::GradientAccumulationSupport::kSupported) {
      OpRegistry::Global()->Register(
          [alg](OpRegistrationData* op_reg_data) -> Status {
            return RegisterPerTableLoadOpsForAlgorithmBody(alg, true,
                                                           op_reg_data);
          });
    }
  }
  // Retrieve ops
  for (tpu::OptimizationAlgorithm alg : tpu::GetOptimizationAlgorithms()) {
    OpRegistry::Global()->Register(
        [alg](OpRegistrationData* op_reg_data) -> Status {
          return RegisterPerTableRetrieveOpsForAlgorithmBody(alg, false,
                                                             op_reg_data);
        });
    tpu::GradientAccumulationSupport grad_accum_support;
    TF_CHECK_OK(GetGradientAccumulationSupport(alg, &grad_accum_support));
    if (grad_accum_support == tpu::GradientAccumulationSupport::kSupported) {
      OpRegistry::Global()->Register(
          [alg](OpRegistrationData* op_reg_data) -> Status {
            return RegisterPerTableRetrieveOpsForAlgorithmBody(alg, true,
                                                               op_reg_data);
          });
    }
  }
}

}  // namespace

REGISTER_OP("RecvTPUEmbeddingActivations")
    .Output("outputs: num_outputs * float32")
    .Attr("num_outputs: int >= 1")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
      string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      tpu::TPUEmbeddingConfiguration config;
      if (!config.ParseFromString(config_string)) {
        return errors::InvalidArgument("Malformed tpu_embedding_config.");
      }
      tpu::AddDefaultEmbeddingOutputLayoutIfNeeded(&config);
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
      return Status::OK();
    })
    .Doc(R"doc(
An op that receives embedding activations on the TPU.

The TPU system performs the embedding lookups and aggregations specified by
the arguments to TPUEmbeddingEnqueue(Integer/Sparse/SparseTensor)Batch. The
results of these aggregations are visible to the Tensorflow Graph as the
outputs of a RecvTPUEmbeddingActivations op. This op returns a list containing
one Tensor of activations per table specified in the model. There can be at
most one RecvTPUEmbeddingActivations op in the TPU graph.

outputs: A TensorList of embedding activations containing one Tensor per
    embedding table in the model.
num_outputs: The number of output activation tensors, equal to the number of
    embedding tables in the model.
config: Serialized TPUEmbeddingConfiguration proto.
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
from the Tensors returned by TPUEmbeddingDequeueActivations. The presence of this
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

      return Status::OK();
    })
    .Doc(R"doc(
An op that performs gradient updates of embedding tables.

The TensorList argument has the same length and shapes as the return value of
TPUEmbeddingReceiveActivations, but contains gradients of the model's loss
with respect to the embedding activations. The embedding tables are updated
from these gradients via the optimizer specified in the configuration given
to tpu.initialize_system.

inputs: A TensorList of gradients with which to update embedding tables.
    It contains one tensor per embedding table in the model.
learning_rates: A list of float32 scalars, one for each embedding table,
    containing the learning rates for each table when dynamic learning rate is
    enabled through the OptimizationParameters in TPUEmbeddingConfiguration.
    When the learning rate is constant, the list should be empty.
config: Serialized TPUEmbeddingConfiguration proto.
)doc");

REGISTER_OP("EnqueueTPUEmbeddingIntegerBatch")
    .Input("batch: N * int32")
    .Input("mode_override: string")
    .Attr("N: int >= 1")
    .Attr("device_ordinal: int = -1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
An op that enqueues a list of input batch tensors to TPUEmbedding.

batch: A list of 1D tensors, one for each embedding table, containing the
    indices into the tables.
mode_override: A string input that overrides the mode specified in the
    TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
    'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
    in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
device_ordinal: The TPU device to use. Should be >= 0 and less than the number
    of TPU cores in the task on which the node is placed.
)doc");

REGISTER_OP("EnqueueTPUEmbeddingSparseBatch")
    .Input("sample_indices: N * int32")
    .Input("embedding_indices: N * int32")
    .Input("aggregation_weights: N * float32")
    .Input("mode_override: string")
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

      return Status::OK();
    })
    .Doc(R"doc(
An op that enqueues TPUEmbedding input indices from a SparseTensor.

This Op eases the porting of code that uses embedding_lookup_sparse(),
although some Python preprocessing of the SparseTensor arguments to
embedding_lookup_sparse() is required to produce the arguments to this Op,
since only a single EnqueueTPUEmbeddingSparseBatch Op is allowed per training
step.

The tensors at corresponding positions in the three input lists
must have the same shape, i.e. rank 1 with dim_size() equal to the total
number of lookups into the table described by the corresponding table_id.

sample_indices: A list of rank 1 Tensors specifying the training example and
    feature to which the corresponding embedding_indices and aggregation_weights
    values belong. sample_indices[i] must equal b * nf + f, where nf is the
    number of features from the corresponding table, f is in [0, nf), and
    b is in [0, batch size).
embedding_indices: A list of rank 1 Tensors, indices into the embedding tables.
aggregation_weights: A list of rank 1 Tensors containing per sample -- i.e. per
    (training example, feature) -- aggregation weights.
mode_override: A string input that overrides the mode specified in the
    TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
    'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
    in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
device_ordinal: The TPU device to use. Should be >= 0 and less than the number
    of TPU cores in the task on which the node is placed.
combiners: A list of string scalars, one for each embedding table that specify
    how to normalize the embedding activations after weighted summation.
    Supported combiners are 'mean', 'sum', or 'sqrtn'. It is invalid to have
    the sum of the weights be 0 for 'mean' or the sum of the squared weights be
    0 for 'sqrtn'. If combiners isn't passed, the default is to use 'sum' for
    all tables.
)doc");

REGISTER_OP("EnqueueTPUEmbeddingSparseTensorBatch")
    .Input("sample_indices: N * int32")
    .Input("embedding_indices: N * int32")
    .Input("aggregation_weights: N * float32")
    .Input("mode_override: string")
    .Attr("N: int >= 1")
    .Attr("device_ordinal: int = -1")
    .Attr("combiners: list(string) = []")
    .Attr("table_ids: list(int)")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
This Op eases the porting of code that uses tf.nn.embedding_lookup_sparse().

sample_indices[i], embedding_indices[i] and aggregation_weights[i] correspond
to the ith feature. table_ids[i] indicates which embedding table to look up ith
feature.

The tensors at corresponding positions in the three input lists (sample_indices,
embedding_indices and aggregation_weights) must have the same shape, i.e. rank 1
with dim_size() equal to the total number of lookups into the table described by
the corresponding feature.

sample_indices: A list of rank 1 Tensors specifying the training example to
    which the corresponding embedding_indices and aggregation_weights values
    belong. It corresponds to sp_ids.indices[:,0] in  embedding_lookup_sparse().
embedding_indices: A list of rank 1 Tensors, indices into the embedding tables.
    It corresponds to sp_ids.values in embedding_lookup_sparse().
aggregation_weights: A list of rank 1 Tensors containing per training example
    aggregation weights. It corresponds to sp_weights.values in
    embedding_lookup_sparse().
mode_override: A string input that overrides the mode specified in the
    TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
    'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
    in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
device_ordinal: The TPU device to use. Should be >= 0 and less than the number
    of TPU cores in the task on which the node is placed.
combiners: A list of string scalars, one for each embedding table that specify
    how to normalize the embedding activations after weighted summation.
    Supported combiners are 'mean', 'sum', or 'sqrtn'. It is invalid to have
    the sum of the weights be 0 for 'mean' or the sum of the squared weights be
    0 for 'sqrtn'. If combiners isn't passed, the default is to use 'sum' for
    all tables.
table_ids: A list of integers specifying the identifier of the embedding table
    (offset of TableDescriptor in the TPUEmbeddingConfiguration) to lookup the
    corresponding input. The ith input is looked up using table_ids[i]. The size
    of the table_ids list must be equal to that of sample_indices,
    embedding_indices and aggregation_weights.
)doc");

}  // namespace tensorflow
