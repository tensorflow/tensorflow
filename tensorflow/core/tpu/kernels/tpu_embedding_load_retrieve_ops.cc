/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/kernels/tpu_embedding_load_retrieve_ops.h"

#include <stddef.h>

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/tpu/optimization_parameters.pb.h"
#include "tensorflow/core/tpu/ops/tpu_embedding_shape_util.h"
#include "tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.h"
#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"

using tensorflow::tpu::TPUEmbeddingConfiguration;
using tensorflow::tpu::TpuEmbeddingShapeUtil;

namespace tensorflow {

// Computes (and VLOGs) the expected shapes of the embedding table shards.
Status ComputeExpectedTableShardShapes(const TPUEmbeddingConfiguration& config,
                                       int shard_id, int num_shards,
                                       const string& op_name,
                                       std::vector<TensorShape>* table_shapes) {
  std::vector<TensorShapeProto> shape_protos;
  const int num_tables = config.table_descriptor_size();
  TF_RETURN_IF_ERROR(TpuEmbeddingShapeUtil::ComputeTableShapes(
      config, shard_id, num_shards, &shape_protos));
  if (num_tables != shape_protos.size()) {
    return errors::InvalidArgument(
        op_name, ": The size of the shape_protos vector ", shape_protos.size(),
        " must be the same as the number of tables ", num_tables);
  }
  for (int table_id = 0; table_id < num_tables; ++table_id) {
    const TensorShape& shape = TensorShape(shape_protos[table_id]);
    table_shapes->push_back(shape);

    const auto& table_descriptor = config.table_descriptor(table_id);
    VLOG(1) << "Table " << table_id << " (name " << table_descriptor.name()
            << ") has shape: " << shape.DebugString()
            << " on shard: " << shard_id << " (of " << num_shards << ").";
  }

  return absl::OkStatus();
}

// Logs min/max/avg for the specified state_variable array.
void LogRangeStatistics(int32 table_id, int32 state_variable_index,
                        absl::Span<const float> state_variable) {
  if (VLOG_IS_ON(5)) {
    float min = std::numeric_limits<float>::infinity();
    float max = -std::numeric_limits<float>::infinity();
    double avg = 0.0;
    for (int elt = 0; elt < state_variable.size(); ++elt) {
      if (state_variable[elt] < min) min = state_variable[elt];
      if (state_variable[elt] > max) max = state_variable[elt];
      avg += state_variable[elt];
    }
    LOG(INFO) << "Table " << table_id << " state_variable "
              << state_variable_index << " min " << min << " max " << max
              << " avg " << avg / state_variable.size() << " total elts "
              << state_variable.size();
  }
}

LoadAllTPUEmbeddingParametersOp::LoadAllTPUEmbeddingParametersOp(
    OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  string config_string;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string));

  OP_REQUIRES(
      ctx, config_.ParseFromString(config_string),
      errors::InvalidArgument("LoadAllTPUEmbeddingParametersOp: Failed to "
                              "parse TPUEmbeddingConfiguration "
                              "proto from config attr"));
  // Auto populate the feature descriptor
  // TODO (b/201806244): remove this logic after the change to the
  // initialization to the config proto.
  OP_REQUIRES_OK(ctx, PopulateMissingFieldsInTPUEmbeddingConfig(&config_));

  int num_shards;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_shards", &num_shards));
  int shard_id;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shard_id", &shard_id));

  OP_REQUIRES_OK(ctx, ComputeExpectedTableShardShapes(
                          config_, shard_id, num_shards,
                          "LoadAllTPUEmbeddingParametersOp", &table_shapes_));
}

void LoadAllTPUEmbeddingParametersOp::GetStateVariables(
    OpKernelContext* ctx,
    std::array<std::vector<absl::Span<const float>>,
               tpu::kMaxAuxiliaryParameterCount + 1>& state_variable_vector) {
  std::array<OpInputList, tpu::kMaxAuxiliaryParameterCount + 1> state_variable;
  OP_REQUIRES_OK(ctx, ctx->input_list("parameters", &state_variable[0]));
  for (int i = 1; i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
    OP_REQUIRES_OK(
        ctx, ctx->input_list(absl::StrCat("auxiliary", i), &state_variable[i]));
  }
  const int num_tables = state_variable[0].size();
  // This should be enforced by Tensorflow's type system.
  for (int i = 1; i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
    CHECK_EQ(num_tables, state_variable[i].size());
  }

  OP_REQUIRES(ctx, num_tables == table_shapes_.size(),
              errors::InvalidArgument(
                  "LoadAllTPUEmbeddingParametersOp has ", num_tables,
                  " inputs in lists but config specifies ",
                  table_shapes_.size(), " embedding tables."));

  CHECK_EQ(num_tables, config_.table_descriptor_size());
  for (int table_id = 0; table_id < num_tables; ++table_id) {
    const auto& table_descriptor = config_.table_descriptor(table_id);
    std::vector<tpu::StateVariableSpecification> state_variable_specs;
    Status status = tpu::GetOptimizationAlgorithmStateVariables(
        table_descriptor.optimization_parameters(), &state_variable_specs);
    OP_REQUIRES(ctx, status.ok(),
                errors::InvalidArgument(
                    "LoadAllTPUEmbeddingParametersOp: No optimization "
                    "algorithm specified for table ",
                    table_id, " (named ", table_descriptor.name(), ")"));
    OP_REQUIRES(
        ctx, state_variable[0][table_id].shape() == table_shapes_[table_id],
        errors::InvalidArgument(
            "LoadAllTPUEmbeddingParametersOp: Embeddings for table ", table_id,
            " (named ", table_descriptor.name(), ") has shape ",
            state_variable[0][table_id].shape().DebugString(),
            " but config specifies table shape ",
            table_shapes_[table_id].DebugString()));
    for (int i = 1; i < state_variable_specs.size(); ++i) {
      OP_REQUIRES(
          ctx, state_variable[i][table_id].shape() == table_shapes_[table_id],
          errors::InvalidArgument(
              "LoadAllTPUEmbeddingParametersOp: Auxiliary ", i - 1,
              " for table ", table_id, " has shape ",
              state_variable[i][table_id].shape().DebugString(),
              " but config specifies table shape ",
              table_shapes_[table_id].DebugString()));
    }
    const int64 num_elements = state_variable[0][table_id].NumElements();
    VLOG(1) << "Table " << table_id << " (name " << table_descriptor.name()
            << ") has shape: " << table_shapes_[table_id].DebugString()
            << ", number of elements: " << num_elements;
    for (int i = 0; i < state_variable_specs.size(); ++i) {
      OP_REQUIRES(ctx,
                  state_variable[i][table_id].NumElements() == num_elements,
                  errors::InvalidArgument(
                      "LoadAllTPUEmbeddingParametersOp: Embeddings/auxiliary ",
                      i, " for table ", table_id, " has element count ",
                      state_variable[i][table_id].NumElements(),
                      " but config requires count ", num_elements));
      const float* state_variable_i_ptr =
          state_variable[i][table_id].flat<float>().data();
      state_variable_vector[i].push_back(absl::MakeConstSpan(
          state_variable_i_ptr, static_cast<size_t>(num_elements)));
      LogRangeStatistics(
          table_id, i, absl::MakeConstSpan(state_variable_i_ptr, num_elements));
    }
    for (int i = state_variable_specs.size();
         i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
      OP_REQUIRES(ctx, state_variable[i][table_id].NumElements() == 0,
                  errors::InvalidArgument(
                      "LoadAllTPUEmbeddingParametersOp: Auxiliary ", i,
                      " for table ", table_id, " has element count ",
                      state_variable[i][table_id].NumElements(),
                      " but config requires empty tensor"));
      state_variable_vector[i].push_back(absl::Span<const float>());
    }
  }
}

void LoadAllTPUEmbeddingParametersOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "LoadAllTPUEmbeddingParameters::Compute";

  std::array<std::vector<absl::Span<const float>>,
             tpu::kMaxAuxiliaryParameterCount + 1>
      state_variable_vector;

  GetStateVariables(ctx, state_variable_vector);
  const int num_tables = state_variable_vector[0].size();

  std::unique_ptr<ApiConverter::TpuEmbeddingEngineParametersData> params =
      ApiConverter::Create(num_tables);
  std::array<std::vector<FloatListRef>, tpu::kMaxAuxiliaryParameterCount + 1>
      params_data;
  for (size_t i = 0; i < tpu::kMaxAuxiliaryParameterCount + 1; i++) {
    params_data[i] = std::vector<FloatListRef>(num_tables);
    for (size_t table_id = 0; table_id < num_tables; table_id++) {
      params->c_params.parameters[i][table_id] = &(params_data[i][table_id]);
      params->c_params.parameters[i][table_id]->size =
          state_variable_vector[i][table_id].size();
      params->c_params.parameters[i][table_id]->ptr =
          const_cast<float*>(state_variable_vector[i][table_id].data());
    }
  }
  StatusHelper status;
  stream_executor::tpu::OpsApiFn()->TpuEmbeddingEngine_WriteParametersFn(
      &(params->c_params), status.c_status);
  OP_REQUIRES_OK(ctx, status.status());

  VLOG(1) << "LoadAllTPUEmbeddingParameters::Compute done";
}

RetrieveAllTPUEmbeddingParametersOp::RetrieveAllTPUEmbeddingParametersOp(
    OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  string config_string;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string));

  OP_REQUIRES(
      ctx, config_.ParseFromString(config_string),
      errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                              "proto from config attr"));

  // Auto populate the feature descriptor
  // TODO (b/201806244): remove this logic after the change to the
  // initialization to the config proto.
  OP_REQUIRES_OK(ctx, PopulateMissingFieldsInTPUEmbeddingConfig(&config_));

  int num_shards;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_shards", &num_shards));
  int shard_id;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shard_id", &shard_id));

  OP_REQUIRES_OK(ctx,
                 ComputeExpectedTableShardShapes(
                     config_, shard_id, num_shards,
                     "RetrieveAllTPUEmbeddingParametersOp", &table_shapes_));
}

void RetrieveAllTPUEmbeddingParametersOp::GetStateVariables(
    OpKernelContext* ctx,
    std::array<std::vector<absl::Span<float>>,
               tpu::kMaxAuxiliaryParameterCount + 1>& state_variable_vector,
    std::vector<int>& num_state_variables) {
  std::array<OpOutputList, tpu::kMaxAuxiliaryParameterCount + 1> state_variable;
  OP_REQUIRES_OK(ctx, ctx->output_list("parameters", &state_variable[0]));
  for (int i = 1; i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
    OP_REQUIRES_OK(ctx, ctx->output_list(absl::StrCat("auxiliary", i),
                                         &state_variable[i]));
  }
  const int num_tables = state_variable[0].size();
  // This should be enforced by Tensorflow's type system.
  for (int i = 1; i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
    CHECK_EQ(num_tables, state_variable[i].size());
  }

  OP_REQUIRES(ctx, num_tables == table_shapes_.size(),
              errors::InvalidArgument(
                  "RetrieveAllTPUEmbeddingParametersOp has ", num_tables,
                  " outputs in lists but config specifies ",
                  table_shapes_.size(), " embedding tables."));

  for (auto& v : state_variable_vector) {
    v.resize(num_tables);
  }
  num_state_variables.resize(num_tables);

  // Get locations to write returned data
  for (int table_id = 0; table_id < num_tables; ++table_id) {
    const auto& table_descriptor = config_.table_descriptor(table_id);

    std::vector<tpu::StateVariableSpecification> state_variable_specs;
    Status status = tpu::GetOptimizationAlgorithmStateVariables(
        table_descriptor.optimization_parameters(), &state_variable_specs);
    OP_REQUIRES(
        ctx, status.ok(),
        errors::InvalidArgument("RetrieveAllTPUEmbeddingParametersOp: No "
                                "optimization algorithm specified for table ",
                                table_id));
    num_state_variables[table_id] = state_variable_specs.size();
    const int64 num_elements = table_shapes_[table_id].num_elements();
    for (int i = 0; i < state_variable_specs.size(); ++i) {
      Tensor* state_variable_tensor;
      OP_REQUIRES_OK(
          ctx, state_variable[i].allocate(table_id, table_shapes_[table_id],
                                          &state_variable_tensor));
      float* state_variable_ptr = state_variable_tensor->flat<float>().data();
      state_variable_vector[i][table_id] =
          absl::MakeSpan(state_variable_ptr, num_elements);
    }
    // Fill in auxiliary values after the number actually used for table_id
    // with empty 2-D tensors.
    for (int i = state_variable_specs.size();
         i <= tpu::kMaxAuxiliaryParameterCount; ++i) {
      Tensor* auxiliary_tensor;
      TensorShape shape;
      std::array<int32, 2> dims = {{0, 0}};
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(dims, &shape));
      OP_REQUIRES_OK(
          ctx, state_variable[i].allocate(table_id, shape, &auxiliary_tensor));
      state_variable_vector[i][table_id] = absl::Span<float>();
    }
  }
}

void RetrieveAllTPUEmbeddingParametersOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "RetrieveAllTPUEmbeddingParameters::Compute";

  std::array<std::vector<absl::Span<float>>,
             tpu::kMaxAuxiliaryParameterCount + 1>
      state_variable_vector;
  std::vector<int> num_state_variables;

  GetStateVariables(ctx, state_variable_vector, num_state_variables);
  const int num_tables = state_variable_vector[0].size();

  std::unique_ptr<ApiConverter::TpuEmbeddingEngineParametersData> params =
      ApiConverter::Create(num_tables);
  std::array<std::vector<FloatListRef>, tpu::kMaxAuxiliaryParameterCount + 1>
      params_data;
  for (size_t i = 0; i < tpu::kMaxAuxiliaryParameterCount + 1; i++) {
    params_data[i] = std::vector<FloatListRef>(num_tables);
    for (size_t table_id = 0; table_id < num_tables; table_id++) {
      params->c_params.parameters[i][table_id] = &(params_data[i][table_id]);
      params->c_params.parameters[i][table_id]->size =
          state_variable_vector[i][table_id].size(),
      params->c_params.parameters[i][table_id]->ptr =
          state_variable_vector[i][table_id].data();
    }
  }
  StatusHelper status;
  stream_executor::tpu::OpsApiFn()->TpuEmbeddingEngine_ReadParametersFn(
      &(params->c_params), status.c_status);
  OP_REQUIRES_OK(ctx, status.status());

  if (VLOG_IS_ON(5)) {
    for (int table_id = 0; table_id < num_tables; ++table_id) {
      for (int i = 0; i < num_state_variables[table_id]; ++i) {
        LogRangeStatistics(table_id, i, state_variable_vector[i][table_id]);
      }
    }
  }
}

#ifdef LIBTPU_ON_GCE

REGISTER_KERNEL_BUILDER(
    Name("LoadAllTPUEmbeddingParameters").Device(DEVICE_CPU),
    LoadAllTPUEmbeddingParametersOp);
REGISTER_KERNEL_BUILDER(
    Name("RetrieveAllTPUEmbeddingParameters").Device(DEVICE_CPU),
    RetrieveAllTPUEmbeddingParametersOp);

#endif  // LIBTPU_ON_GCE
}  // namespace tensorflow
