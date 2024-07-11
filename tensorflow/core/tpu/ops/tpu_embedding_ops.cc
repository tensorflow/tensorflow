/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/ops/tpu_embedding_ops.h"

#include <array>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/status_macros.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/protobuf/tpu/optimization_parameters.pb.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/ops/tpu_embedding_shape_util.h"
#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"
#include "tensorflow/core/tpu/tpu_embedding_output_layout_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using tensorflow::tpu::TPUEmbeddingConfiguration;

REGISTER_OP("ExecuteTPUEmbeddingPartitioner")
    .Output("common_config: string")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      std::string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      TPUEmbeddingConfiguration config;
      TF_RET_CHECK(config.ParseFromString(config_string));
      if (config.mode() == TPUEmbeddingConfiguration::UNSPECIFIED) {
        return absl::InvalidArgumentError(
            "TPUEmbeddingConfiguration.mode is INVALID.  Must be INFERENCE, "
            "TRAINING, or BACKWARD_PASS_ONLY");
      }
      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("ConfigureTPUEmbeddingMemory")
    .Input("common_config: string")
    .Output("memory_config: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      TF_RET_CHECK(c->num_inputs() == 1);
      // Validate that all the input shape is compatible.
      ShapeHandle input(c->Scalar());
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), input, &input));
      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("CollateTPUEmbeddingMemory")
    .Input("memory_configs: N * string")
    .Output("merged_memory_config: string")
    .Attr("N: int >= 1")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      TF_RET_CHECK(c->num_inputs() > 0);
      ShapeHandle input(c->Scalar());
      // Validate that all the inputs are compatible with the correct
      // vector shape.
      for (int i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->Merge(c->input(i), input, &input));
      }
      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("ConfigureTPUEmbeddingHost")
    .Input("common_config: string")
    .Input("memory_config: string")
    .Output("network_config: string")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      std::string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      TPUEmbeddingConfiguration config;
      TF_RET_CHECK(config.ParseFromString(config_string));
      if (config.mode() == TPUEmbeddingConfiguration::UNSPECIFIED) {
        return absl::InvalidArgumentError(
            "TPUEmbeddingConfiguration.mode is INVALID.  Must be INFERENCE, "
            "TRAINING, or BACKWARD_PASS_ONLY");
      }
      TF_RET_CHECK(c->num_inputs() == 2);
      ShapeHandle input(c->Scalar());
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), input, &input));
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), input, &input));
      c->set_output(0, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("ConnectTPUEmbeddingHosts")
    .Input("network_configs: N * string")
    .Attr("N: int >= 1")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      TF_RET_CHECK(c->num_inputs() > 0);
      ShapeHandle input(c->Scalar());
      // Validate that all the inputs are compatible with the correct
      // vector shape.
      for (int i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->Merge(c->input(i), input, &input));
      }
      return absl::OkStatus();
    });

REGISTER_OP("FinalizeTPUEmbedding")
    .Input("common_config: string")
    .Input("memory_config: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      // Validate that all the inputs are compatible with the correct
      // vector shape.
      TF_RET_CHECK(c->num_inputs() == 2);
      ShapeHandle input(c->Scalar());
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), input, &input));
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), input, &input));
      return absl::OkStatus();
    });

// After configuring the TPU system (detailed in tpu_configuration_ops.cc),
// you may, if desired, run ExecuteTPUEmbeddingPartitioner,
// ConfigureTPUEmbeddingHost, _ConnectInterTPUEmbeddingCommunication and
// FinalizeTPUEmbedding Ops to configure the TPUEmbeddings.
//
// 1) The ExecuteTPUEmbeddingPartitioner Op runs on TPU_SYSTEM of task 0. It
//    runs the embedding layer partitioner and computes the HBM size (in bytes)
//    needed for TPUEmbedding operation. Note that this Op does not need to wait
//    until the entire TPU system is configured, rather it only needs to wait
//    till _ConfigureDistributedTPU completes. Also stores the HBM size and the
//    embedding partitioner output in the system metadata where it can be used
//    while compiling embedding Ops for TPU.
// 2) The ConfigureTPUEmbeddingMemory Op runs on the TPU:0's host device of all
//    tasks. Using the output of the ExecuteTPUEmbeddingPartitioner Op, it
//    allocates HBM memory, initializes the TPUEmbeddingManager and store the
//    HBM buffer configuration.
// 3) The ConfigureTPUEmbeddingHost Op runs on the TPU:0's host device of all
//    tasks. Using the output of the ExecuteTPUEmbeddingPartitioner Op, it
//    builds the program that executes on the TPUEmbeddings, configures the
//    TPUEmbedding hardware and sets up the TPUEmbedding host software. Using
//    the output of the ConfigureTPUEmbeddingMemory Op that it receives from all
//    tasks, it checks that HBM segment sizes are equal, and combines each
//    task's allocation info to create a global map of HBM base addresses. It
//    uses that to initialize the TPUEmbeddingManager, and also provides the
//    hostname:port for inter-TPUEmbedding agreement of minibatch sizing.
// 4) The _ConnectInterTPUEmbeddingCommunication Op runs on the TPU:0's host
//    device of all tasks. It uses the hostname:port output from all
//    ConfigureTPUEmbeddingHost Ops to form all-to-all connections between all
//    tasks for inter-TPUEmbedding agreement.
// 5) The FinalizeTPUEmbedding Op runs on TPU_SYSTEM of
//    task 0. It takes as input the outputs from all ConfigureTPUEmbeddingHost
//    Ops and validates that the HBM base address (in bytes) used for
//    TPUEmbedding operation is the same. Also stores the common HBM base
//    address in the system metadata where it can be used while compiling
//    embedding Ops.

std::vector<std::string> GetPerTableLoadOptimizationParametersOps() {
  VLOG(1) << "GetPerTableLoadOptimizationParametersOps ";
  std::vector<std::string> result;
  for (tpu::OptimizationAlgorithm alg : tpu::GetOptimizationAlgorithms()) {
    const auto alg_name = tpu::GetOptimizationAlgorithmName(alg);
    const auto op_name =
        absl::StrCat("LoadTPUEmbedding", alg_name, "Parameters");
    result.push_back(op_name);
    // Some of these don't actually exist. Specifically
    // 'CenteredRMSProp.*GradAccumDebug' and
    // 'MDLAdagradLight.*GradAccumDebug''. Its ok to include them here as they
    // are only used to check for ops during re-write passes.
    const auto op_name_debug =
        absl::StrCat("LoadTPUEmbedding", alg_name, "ParametersGradAccumDebug");
    result.push_back(op_name_debug);
  }
  return result;
}

std::vector<std::string> GetPerTableRetrieveOptimizationParametersOps() {
  VLOG(1) << "GetPerTableRetrieveOptimizationParametersOps ";
  std::vector<std::string> result;
  for (tpu::OptimizationAlgorithm alg : tpu::GetOptimizationAlgorithms()) {
    const auto alg_name = tpu::GetOptimizationAlgorithmName(alg);
    const auto op_name =
        absl::StrCat("RetrieveTPUEmbedding", alg_name, "Parameters");
    result.push_back(op_name);
    // Some of these don't actually exist. Specifically
    // 'CenteredRMSProp.*GradAccumDebug' and
    // 'MDLAdagradLight.*GradAccumDebug'. Its ok to include them here as they
    // are only used to check for ops during re-write passes.
    const auto op_name_debug = absl::StrCat("RetrieveTPUEmbedding", alg_name,
                                            "ParametersGradAccumDebug");
    result.push_back(op_name_debug);
  }
  return result;
}

static_assert(tpu::kMaxAuxiliaryParameterCount == 7,
              "Need to update parameter count in "
              "LoadAllTPUEmbeddingParameters and "
              "RetrieveAllTPUEmbeddingParameters ops if "
              "kMaxAuxiliaryParameterCount changes");

REGISTER_OP("LoadAllTPUEmbeddingParameters")
    .Input("parameters: NumTables * float")
    .Input("auxiliary1: NumTables * float")
    .Input("auxiliary2: NumTables * float")
    .Input("auxiliary3: NumTables * float")
    .Input("auxiliary4: NumTables * float")
    .Input("auxiliary5: NumTables * float")
    .Input("auxiliary6: NumTables * float")
    .Input("auxiliary7: NumTables * float")
    .Attr("NumTables: int")
    .Attr("config: string")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      std::string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      TPUEmbeddingConfiguration config;
      TF_RET_CHECK(config.ParseFromString(config_string));
      int num_shards;
      TF_RETURN_IF_ERROR(c->GetAttr("num_shards", &num_shards));
      int shard_id;
      TF_RETURN_IF_ERROR(c->GetAttr("shard_id", &shard_id));
      std::vector<TensorShapeProto> table_shapes;
      TF_RETURN_IF_ERROR(tpu::TpuEmbeddingShapeUtil::ComputeTableShapes(
          config, shard_id, num_shards, &table_shapes));
      std::array<std::vector<ShapeHandle>, tpu::kMaxAuxiliaryParameterCount + 1>
          accumulators;
      TF_RETURN_IF_ERROR(c->input("parameters", &accumulators[0]));
      for (int i = 1; i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
        TF_RETURN_IF_ERROR(
            c->input(absl::StrCat("auxiliary", i), &accumulators[i]));
      }
      TF_RET_CHECK(accumulators[0].size() == table_shapes.size());
      // This should be enforced by Tensorflow's type system.
      for (int i = 1; i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
        CHECK_EQ(accumulators[0].size(), accumulators[i].size());  // Crash OK
      }
      for (int table_id = 0; table_id < accumulators[0].size(); ++table_id) {
        // Verify shapes have rank 2 and are compatible when they are required
        // to be valid.
        ShapeHandle parameter_shape;
        TF_RETURN_IF_ERROR(
            c->WithRank(accumulators[0][table_id], 2, &parameter_shape));

        std::vector<tpu::StateVariableSpecification> state_variable_specs;
        absl::Status status = tpu::GetOptimizationAlgorithmStateVariables(
            config.table_descriptor(table_id).optimization_parameters(),
            &state_variable_specs);
        TF_RET_CHECK(status.ok());

        for (int i = 1; i < state_variable_specs.size(); ++i) {
          ShapeHandle accumulator_i_shape;
          TF_RETURN_IF_ERROR(
              c->WithRank(accumulators[i][table_id], 2, &accumulator_i_shape));
          ShapeHandle merged;
          TF_RETURN_IF_ERROR(c->Merge(accumulators[0][table_id],
                                      accumulator_i_shape, &merged));
          // Verify shapes are compatible with the shapes specified in
          // the config.
          ShapeHandle from_config;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromShapeProto(table_shapes[table_id], &from_config));
          ShapeHandle merged_with_config;
          TF_RETURN_IF_ERROR(
              c->Merge(merged, from_config, &merged_with_config));
        }
        // Ensure that other state variables are empty to catch bugs in
        // CombineTPUEmbeddingLoadRetrievePass output or manually-written
        // equivalent code.
        for (int i = state_variable_specs.size();
             i < tpu::kMaxAuxiliaryParameterCount + 1; ++i) {
          ShapeHandle accumulator_i_shape;
          TF_RETURN_IF_ERROR(c->WithRankAtLeast(accumulators[i][table_id], 1,
                                                &accumulator_i_shape));
          shape_inference::DimensionHandle dim;
          TF_RETURN_IF_ERROR(
              c->WithValue(c->NumElements(accumulator_i_shape), 0, &dim));
        }
      }
      return absl::OkStatus();
    });

REGISTER_OP("RetrieveAllTPUEmbeddingParameters")
    .Output("parameters: NumTables * float")
    .Output("auxiliary1: NumTables * float")
    .Output("auxiliary2: NumTables * float")
    .Output("auxiliary3: NumTables * float")
    .Output("auxiliary4: NumTables * float")
    .Output("auxiliary5: NumTables * float")
    .Output("auxiliary6: NumTables * float")
    .Output("auxiliary7: NumTables * float")
    .Attr("NumTables: int")
    .Attr("config: string")
    .Attr("num_shards: int")
    .Attr("shard_id: int")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      std::string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      TPUEmbeddingConfiguration config;
      TF_RET_CHECK(config.ParseFromString(config_string));
      int num_shards;
      TF_RETURN_IF_ERROR(c->GetAttr("num_shards", &num_shards));
      int shard_id;
      TF_RETURN_IF_ERROR(c->GetAttr("shard_id", &shard_id));
      std::vector<TensorShapeProto> table_shapes;
      TF_RETURN_IF_ERROR(tpu::TpuEmbeddingShapeUtil::ComputeTableShapes(
          config, shard_id, num_shards, &table_shapes));
      int num_tables;
      TF_RETURN_IF_ERROR(c->GetAttr("NumTables", &num_tables));
      TF_RET_CHECK(num_tables == table_shapes.size());
      TF_RET_CHECK(num_tables == config.table_descriptor_size());
      for (int i = 0; i < tpu::kMaxAuxiliaryParameterCount + 1; ++i) {
        std::vector<ShapeHandle> output_handles;
        for (int table_id = 0; table_id < table_shapes.size(); ++table_id) {
          std::vector<tpu::StateVariableSpecification> state_variable_specs;
          TF_RETURN_IF_ERROR(tpu::GetOptimizationAlgorithmStateVariables(
              config.table_descriptor(table_id).optimization_parameters(),
              &state_variable_specs));

          TensorShapeProto output_shape_proto =
              (i < state_variable_specs.size()
                   ? table_shapes[table_id]
                   : tpu::TpuEmbeddingShapeUtil::MakeEmpty2DShape());
          ShapeHandle output_shape;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromShapeProto(output_shape_proto, &output_shape));
          output_handles.push_back(output_shape);
        }
        if (i == 0) {
          TF_RETURN_IF_ERROR(c->set_output("parameters", output_handles));
        } else {
          TF_RETURN_IF_ERROR(
              c->set_output(absl::StrCat("auxiliary", i), output_handles));
        }
      }
      return absl::OkStatus();
    });

REGISTER_OP("EnqueueTPUEmbeddingBatch")
    .Input("batch: N * string")
    .Input("mode_override: string")
    .Attr("N: int")
    .Attr("device_ordinal: int = -1")
    .Attr("combiners: list(string) = []")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      std::vector<std::string> combiners;
      TF_RETURN_IF_ERROR(c->GetAttr("combiners", &combiners));
      int n;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
      if (!combiners.empty() && combiners.size() != n) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid length of combiners. Have ", combiners.size(),
                         " but expected 0 or ", n));
      }

      return absl::OkStatus();
    });

REGISTER_OP("XlaRecvTPUEmbeddingActivations")
    .Input("deduplication_data: variant")
    .Output("outputs: num_tables * float32")
    .Attr("num_tables: int >= 1")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int num_tables;
      TF_RETURN_IF_ERROR(c->GetAttr("num_tables", &num_tables));
      if (c->num_outputs() != num_tables) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Number of outputs: %d of the XlaRecvTPUEmbeddingActivations node "
            "does not match the num_tables attribute: %d.",
            c->num_outputs(), num_tables));
      }
      std::string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      tpu::TPUEmbeddingConfiguration config;
      if (!config.ParseFromString(config_string)) {
        return absl::InvalidArgumentError(
            "Malformed config attribute in the XlaRecvTPUEmbeddingActivations "
            "node.");
      }
      std::vector<TensorShapeProto> output_shapes;
      TF_RETURN_IF_ERROR(ComputeOutputTensorShapes(config, &output_shapes));
      if (c->num_outputs() != output_shapes.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Number of outputs: %d of the XlaRecvTPUEmbeddingActivations node "
            "does not match the number of tables or features in the TPU "
            "embedding config: %d.",
            c->num_outputs(), output_shapes.size()));
      }
      for (int i = 0; i < c->num_outputs(); ++i) {
        shape_inference::ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromShapeProto(output_shapes[i], &output_shape));
        c->set_output(i, output_shape);
      }
      return absl::OkStatus();
    });

REGISTER_OP("XlaSendTPUEmbeddingGradients")
    .Input("gradients: NumTables * float32")
    .Input("learning_rates: NumLearningRateTags * float32")
    .Input("deduplication_data: variant")
    .Attr("NumTables: int >= 1")
    .Attr("NumLearningRateTags: int >= 0 = 0")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int learning_rate_tag_count;
      TF_RETURN_IF_ERROR(
          c->GetAttr("NumLearningRateTags", &learning_rate_tag_count));
      std::vector<shape_inference::ShapeHandle> learning_rates;
      TF_RETURN_IF_ERROR(c->input("learning_rates", &learning_rates));
      for (int i = 0; i < learning_rate_tag_count; ++i) {
        // Verify that each learning_rates element is scalar
        shape_inference::ShapeHandle learning_rates_shape;
        TF_RETURN_IF_ERROR(
            c->WithRank(learning_rates[i], 0, &learning_rates_shape));
      }

      return absl::OkStatus();
    });

REGISTER_OP("XlaRecvTPUEmbeddingDeduplicationData")
    .Output("output: variant")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

// XlaRecvTPUEmbeddingDeduplicationData returns `output` of an XLA tuple, which
// consists of integer and floating point values. For cases that users needs
// static shape output, this XLA tuple can not be returned. Therefore we create
// a pair of conversion operations, to convert deduplication data (XLA tuple)
// to tensors and vice versa.
// `SplitDedupData` is to split deduplication data XLA tuple into integer and
// floating point tensors. Here we assume deduplication data XLA tuple only
// has two type of elements. We infer output shapes of these two tensors with
// `tuple_mask`, which is a serialized proto of 2-D int tensor. The first column
// of `tuple_mask` is consisted by 0 and 1, where 0 means integer type, 1 means
// floating point type. The second column is length of span, summation of these
// spans should be equal to number of elements in deduplication data XLA tuple.
// For example, `tuple_mask` of tuple (1, 2, 0.1, 3) is [[0, 2], [1, 1], [0, 1]]

REGISTER_OP("SplitDedupData")
    .Input("input: variant")
    .Output("integer_tensor: integer_type")
    .Output("float_tensor: float_type")
    .Attr("integer_type: {int32, int64, uint32, uint64}")
    .Attr("float_type: {half, bfloat16, float}")
    .Attr("tuple_mask: string")
    .Attr("config: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      std::string tuple_mask_str;
      TF_RETURN_IF_ERROR(c->GetAttr("tuple_mask", &tuple_mask_str));

      tensorflow::TensorProto tuple_mask_tensor;
      if (!tuple_mask_tensor.ParseFromString(tuple_mask_str)) {
        return absl::InvalidArgumentError(
            "Malformed `tuple_mask` attr in SplitDedupData Op.");
      }
      const tensorflow::TensorShapeProto& tuple_tensor_shape =
          tuple_mask_tensor.tensor_shape();
      const int num_tuple_elements = tuple_tensor_shape.dim(0).size();
      if (num_tuple_elements == 0) {
        c->set_output(0, c->MakeShape({c->MakeDim(0)}));
        c->set_output(1, c->MakeShape({c->MakeDim(0)}));
        return absl::OkStatus();
      }

      const int tuple_mask_rank = tuple_tensor_shape.dim_size();
      if (tuple_mask_rank != 2) {
        return absl::InvalidArgumentError(absl::StrCat(
            "`tuple_mask` TensorProto must be a rank-2 tensor, but get ",
            tuple_mask_rank));
      }
      TF_RET_CHECK(tuple_mask_tensor.int_val_size() == 2 * num_tuple_elements);

      int integer_offset = 0;  // Offset of integer elements in tuple.
      int float_offset = 0;    // Offset of floating elements in tuple.
      for (int i = 0; i < num_tuple_elements; i++) {
        const int element_type = tuple_mask_tensor.int_val(2 * i);
        const int span_size = tuple_mask_tensor.int_val(2 * i + 1);

        if (element_type == DedupTupleElementType::kInteger) {
          integer_offset += span_size;
        } else if (element_type == DedupTupleElementType::kFloat) {
          float_offset += span_size;
        } else {
          return absl::InvalidArgumentError(absl::StrCat(
              "Unexpected type of element in deduplication tuple, enum = ",
              element_type, ", which is not integer or floating."));
        }
      }

      std::string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      if (!config_string.empty()) {
        tpu::TPUEmbeddingConfiguration config;
        if (!config.ParseFromString(config_string)) {
          return absl::InvalidArgumentError(
              "Malformed config attribute in the SplitDedupData node.");
        }
      }

      const shape_inference::DimensionHandle integer_tensor_dim =
          c->MakeDim(integer_offset);
      const shape_inference::DimensionHandle float_tensor_dim =
          c->MakeDim(float_offset);
      c->set_output(0, c->MakeShape({integer_tensor_dim}));
      c->set_output(1, c->MakeShape({float_tensor_dim}));
      return absl::OkStatus();
    });

// `MergeDedupData` is to merge outputs of `SplitDedupData` back to an XLA tuple
// as deduplication data, with respect to `tuple_mask`.

REGISTER_OP("MergeDedupData")
    .Input("integer_tensor: integer_type")
    .Input("float_tensor: float_type")
    .Output("output: variant")
    .Attr("tuple_mask: string")
    .Attr("integer_type: {int32, int64, uint32, uint64}")
    .Attr("float_type: {half, bfloat16, float}")
    .Attr("config: string = ''")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("ComputeDedupDataSize")
    .Output("num_elements: int32")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("ComputeDedupDataTupleMask")
    .Output("output_shape: int32")
    .Attr("config: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(2));
      return absl::OkStatus();
    });

REGISTER_OP("XlaRecvTPUEmbeddingActivationsV2")
    .Input("deduplication_data: variant")
    .Output("outputs: num_tables * float32")
    .Attr("num_tables: int >= 1")
    .Attr("config: string")
    .Attr("embedding_partitions: string")
    .Attr("hbm_buffers_config: string")
    .Attr("tpu_topology: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int num_tables;
      TF_RETURN_IF_ERROR(c->GetAttr("num_tables", &num_tables));
      if (c->num_outputs() != num_tables) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Number of outputs: %d of the "
                            "XlaRecvTPUEmbeddingActivationsV2 node "
                            "does not match the num_tables attribute: %d.",
                            c->num_outputs(), num_tables));
      }
      std::string config_string;
      TF_RETURN_IF_ERROR(c->GetAttr("config", &config_string));
      tpu::TPUEmbeddingConfiguration config;
      if (!config.ParseFromString(config_string)) {
        return absl::InvalidArgumentError(
            "Malformed config attribute in the "
            "XlaRecvTPUEmbeddingActivationsV2 "
            "node.");
      }
      std::string embedding_partitions_string;
      TF_RETURN_IF_ERROR(
          c->GetAttr("embedding_partitions", &embedding_partitions_string));
      std::string hbm_buffers_config_string;
      TF_RETURN_IF_ERROR(
          c->GetAttr("hbm_buffers_config", &hbm_buffers_config_string));
      std::string tpu_topology_string;
      TF_RETURN_IF_ERROR(c->GetAttr("tpu_topology", &tpu_topology_string));
      std::vector<TensorShapeProto> output_shapes;
      TF_RETURN_IF_ERROR(ComputeOutputTensorShapes(config, &output_shapes));
      if (c->num_outputs() != output_shapes.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Number of outputs: %d of the XlaRecvTPUEmbeddingActivationsV2 "
            "node "
            "does not match the number of tables or features in the TPU "
            "embedding config: %d.",
            c->num_outputs(), output_shapes.size()));
      }
      for (int i = 0; i < c->num_outputs(); ++i) {
        shape_inference::ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromShapeProto(output_shapes[i], &output_shape));
        c->set_output(i, output_shape);
      }
      return absl::OkStatus();
    });

REGISTER_OP("XlaRecvTPUEmbeddingDeduplicationDataV2")
    .Output("output: variant")
    .Attr("config: string")
    .Attr("embedding_partitions: string")
    .Attr("hbm_buffers_config: string")
    .Attr("tpu_topology: string")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("XlaSendTPUEmbeddingGradientsV2")
    .Input("gradients: NumTables * float32")
    .Input("learning_rates: NumLearningRateTags * float32")
    .Input("deduplication_data: variant")
    .Attr("NumTables: int >= 1")
    .Attr("NumLearningRateTags: int >= 0 = 0")
    .Attr("config: string")
    .Attr("embedding_partitions: string")
    .Attr("hbm_buffers_config: string")
    .Attr("tpu_topology: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int learning_rate_tag_count;
      TF_RETURN_IF_ERROR(
          c->GetAttr("NumLearningRateTags", &learning_rate_tag_count));
      std::vector<shape_inference::ShapeHandle> learning_rates;
      TF_RETURN_IF_ERROR(c->input("learning_rates", &learning_rates));
      for (int i = 0; i < learning_rate_tag_count; ++i) {
        // Verify that each learning_rates element is scalar
        shape_inference::ShapeHandle learning_rates_shape;
        TF_RETURN_IF_ERROR(
            c->WithRank(learning_rates[i], 0, &learning_rates_shape));
      }

      return absl::OkStatus();
    });

REGISTER_OP("ComputeDedupDataSizeV2")
    .Output("num_elements: int32")
    .Attr("config: string")
    .Attr("embedding_partitions: string")
    .Attr("hbm_buffers_config: string")
    .Attr("tpu_topology: string")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("ComputeDedupDataTupleMaskV2")
    .Output("output_shape: int32")
    .Attr("config: string")
    .Attr("embedding_partitions: string")
    .Attr("hbm_buffers_config: string")
    .Attr("tpu_topology: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(2));
      return absl::OkStatus();
    });

REGISTER_OP("FinalizeTPUEmbeddingV2")
    .Input("common_config: string")
    .Input("memory_config: string")
    .Output("embedding_partitions: string")
    .Output("hbm_buffers_config: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) -> absl::Status {
      // Validate that all the inputs are compatible with the correct
      // vector shape.
      TF_RET_CHECK(c->num_inputs() == 2);
      ShapeHandle input(c->Scalar());
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), input, &input));
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), input, &input));
      TF_RET_CHECK(c->num_outputs() == 2);
      return absl::OkStatus();
    });

REGISTER_OP("GetTpuTaskId")
    .Output("tpu_task_id: int32")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("UpdateTaskIdAndGlobalCoreArray")
    .Input("tpu_task_id_to_shard_id: task_count * int32")
    .Attr("task_count: int >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow
