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

#ifdef LIBTPU_ON_GCE

#include <string>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"

using ::tensorflow::tpu::TPUEmbeddingConfiguration;

namespace tensorflow {

namespace {
namespace se_tpu = ::stream_executor::tpu;
}

// The ExecuteTPUEmbeddingPartitioner Op is used to run the TPUEmbedding
// partitioner as well as calculate the HBM size (in bytes) required for
// TPUEmbedding operation. It takes as input a TPUEmbeddingConfiguration proto
// which describes all the embedding tables and metadata. It should be run on
// the TPU_SYSTEM device on only one task (by convention, task 0).
// Note that the _ConfigureDistributedTPU Op must have run before this Op so
// that the TpuTopology is added to the TpuMeshCommonState. Subsequent
// TPUEmbedding host configuration Ops (one per task) will use the output of
// this Op.

class ExecuteTPUEmbeddingPartitionerOp : public OpKernel {
 public:
  explicit ExecuteTPUEmbeddingPartitionerOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES(
        ctx, TPUEmbeddingConfiguration().ParseFromString(config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "ExecuteTPUEmbeddingPartitioner::Compute";
    TpuEmbeddingEngine_ExecutePartitioner_Params params;
    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();
    ResourceMgr* rm = GetTPUConfigResourceMgr();
    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);
    params.tpu_mesh_state = mesh_state->data();

    char* common_config_output = nullptr;
    auto cleanup = absl::MakeCleanup([&common_config_output]() {
      tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(
          common_config_output);
    });
    size_t common_config_output_size;
    params.common_config_size = &common_config_output_size;
    params.common_config = &common_config_output;

    StatusHelper status;
    params.status = status.c_status;

    tpu::OpsApiFn()->TpuEmbeddingEngine_ExecutePartitionerFn(&params);
    if (!status.ok()) {
      VLOG(0) << "ExecuteTPUEmbeddingPartitioner::Compute failed"
              << status.status().ToString();
      return;
    }
    std::string common_config_string =
        std::string(common_config_output, common_config_output_size);
    Tensor* output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("common_config", TensorShape({}), &output));
    output->flat<tstring>()(0) = common_config_string;
    VLOG(1) << "ExecuteTPUEmbeddingPartitioner::Compute done";
  }

 private:
  // The embedding layer configuration for the TPUEmbedding host software.
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExecuteTPUEmbeddingPartitionerOp);
};

// Initialize the HBM memory addresses and segments on each host.
// The ConfigureTPUEmbeddingMemoryOp allocates HBM memory used by TPUEmbedding.
// It takes as input a TPUEmbeddingConfiguration proto, which describes all
// the embedding tables, and the output of the
// _ExecuteTPUEmbeddingPartitioner Op. It should be run on one TPU device
// on each task, by convention TPU:0.
class ConfigureTPUEmbeddingMemoryOp : public OpKernel {
 public:
  explicit ConfigureTPUEmbeddingMemoryOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES(
        ctx, TPUEmbeddingConfiguration().ParseFromString(config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "ConfigureTPUEmbeddingMemoryOp::Compute";
    std::string common_config_string = ctx->input(0).flat<tstring>()(0);
    std::string host_config;

    TpuEmbeddingEngine_ConfigureMemory_Params params;
    params.common_config_string = common_config_string.c_str();
    params.common_config_string_size = common_config_string.size();

    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();

    StatusHelper status;
    params.status = status.c_status;

    char* task_host_config_output = nullptr;
    auto task_host_config_cleanup =
        absl::MakeCleanup([&task_host_config_output]() {
          tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(
              task_host_config_output);
        });
    size_t task_host_config_output_size;
    params.task_host_config_size = &task_host_config_output_size;
    params.task_host_config = &task_host_config_output;
    params.num_inputs = ctx->num_inputs();

    tpu::OpsApiFn()->TpuEmbeddingEngine_ConfigureMemoryFn(&params);
    OP_REQUIRES_OK(ctx, status.status());
    std::string task_host_config =
        std::string(task_host_config_output, task_host_config_output_size);

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("task_host_config",
                                             TensorShape({}), &output));
    output->flat<tstring>()(0) = task_host_config;

    VLOG(1) << "ConfigureTPUEmbeddingMemoryOp::Compute done";
  }

  ~ConfigureTPUEmbeddingMemoryOp() override {}

 private:
  // The embedding layer configuration for the TPUEmbedding host software.
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConfigureTPUEmbeddingMemoryOp);
};

// The ConfigureTPUEmbeddingHost Op is used to set up the TPUEmbedding
// software on a given host. It takes as input a TPUEmbeddingConfiguration
// proto which describes all the embedding tables as well as the output of
// the _ExecuteTPUEmbeddingPartitioner Op. It should be run on one TPU device
// on each task, by convention TPU:0.
class ConfigureTPUEmbeddingHostOp : public OpKernel {
 public:
  explicit ConfigureTPUEmbeddingHostOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES(ctx, ctx->num_inputs() > 0,
                errors::InvalidArgument("ConfigureTPUEmbeddingHostOp must "
                                        "have at least one input"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES(
        ctx, TPUEmbeddingConfiguration().ParseFromString(config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "ConfigureTPUEmbeddingHostOp::Compute";
    std::string common_config_string = ctx->input(0).flat<tstring>()(0);

    // Retrieve per-task config received from each
    // ConfigureTPUEmbeddingMemoryOp node.
    OpInputList task_host_config;
    OP_REQUIRES_OK(ctx, ctx->input_list("task_host_config", &task_host_config));

    std::vector<std::string> task_host_config_string(task_host_config.size());
    std::vector<se_tpu::SerializedProto> task_hosts_config(
        task_host_config.size());
    for (int i = 0; i < task_host_config.size(); ++i) {
      task_host_config_string[i] = task_host_config[i].flat<tstring>()(0);
      task_hosts_config[i].bytes = task_host_config_string[i].c_str();
      task_hosts_config[i].size = task_host_config_string[i].size();
    }

    TpuEmbeddingEngine_ConfigureHost_Params params;
    params.common_config_string = common_config_string.c_str();
    params.common_config_string_size = common_config_string.size();

    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();

    char* host_config_output = nullptr;
    auto cleanup = absl::MakeCleanup([&host_config_output]() {
      tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(host_config_output);
    });

    size_t host_config_output_size;
    params.host_config_size = &host_config_output_size;
    params.host_config = &host_config_output;
    params.num_inputs = ctx->num_inputs();

    params.task_host_config = task_hosts_config.data();
    params.task_host_config_size = task_hosts_config.size();

    StatusHelper status;
    params.status = status.c_status;

    tpu::OpsApiFn()->TpuEmbeddingEngine_ConfigureHostFn(&params);

    OP_REQUIRES_OK(ctx, status.status());
    std::string host_config =
        std::string(host_config_output, host_config_output_size);

    Tensor* output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("host_config", TensorShape({}), &output));
    output->flat<tstring>()(0) = host_config;
    VLOG(1) << "ConfigureTPUEmbeddingHostOp::Compute done";
  }

  ~ConfigureTPUEmbeddingHostOp() override {}

 private:
  // The embedding layer configuration for the TPUEmbedding host software.
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConfigureTPUEmbeddingHostOp);
};

// The ConnectInterTPUEmbeddingCommunication op is used to set up gRPC
// connections between instances of the TPUEmbedding host software on different
// hosts; it must be run after ConfigureTPUEmbeddingHostOp has been called on
// each host. It takes as input a string from each host which describes metadata
// about the TPUEmbedding configuration on that host. It should be run on one
// TPU device in the host, by convention TPU:0.
class ConnectInterTPUEmbeddingCommunicationOp : public OpKernel {
 public:
  explicit ConnectInterTPUEmbeddingCommunicationOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES(
        ctx, ctx->num_inputs() > 0,
        errors::InvalidArgument("ConnectInterTPUEmbeddingCommunication must "
                                "have at least one input"));
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "ConnectInterTPUEmbeddingCommunication::Compute";

    std::vector<std::string> hosts_config_string(ctx->num_inputs());
    std::vector<se_tpu::SerializedProto> hosts_config(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      hosts_config_string[i] = ctx->input(i).flat<tstring>()(0);
      hosts_config[i].bytes = hosts_config_string[i].c_str();
      hosts_config[i].size = hosts_config_string[i].size();
    }

    TpuEmbeddingEngine_ConfigureCommunication_Params params;
    StatusHelper status;
    params.status = status.c_status;

    params.host_config = hosts_config.data();
    params.host_config_size = hosts_config.size();

    tpu::OpsApiFn()->TpuEmbeddingEngine_ConfigureCommunicationFn(&params);
    OP_REQUIRES_OK(ctx, status.status());

    VLOG(1) << "ConnectInterTPUEmbeddingCommunication::Compute done";
  }

  ~ConnectInterTPUEmbeddingCommunicationOp() override {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConnectInterTPUEmbeddingCommunicationOp);
};

// The FinalizeTPUEmbeddingSystemConfiguration op is used to configure the
// system once ConfigureTPUEmbeddingHostOp has been called on each host. It
// takes as input a string from each host which describes metadata about the
// TPUEmbedding configuration on that host. It must be run on the TPU system
// device to which the TPUEmbedding hosts are attached.
class FinalizeTPUEmbeddingSystemConfigurationOp : public OpKernel {
 public:
  explicit FinalizeTPUEmbeddingSystemConfigurationOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES(
        ctx, ctx->num_inputs() > 0,
        errors::InvalidArgument("FinalizeTPUEmbeddingSystemConfiguration must "
                                "have at least one input"));
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "FinalizeTPUEmbeddingSystemConfiguration::Compute";

    std::vector<std::string> hosts_config_string(ctx->num_inputs());
    std::vector<se_tpu::SerializedProto> hosts_config(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      hosts_config_string[i] = ctx->input(i).flat<tstring>()(0);
      hosts_config[i].bytes = hosts_config_string[i].c_str();
      hosts_config[i].size = hosts_config_string[i].size();
    }

    TpuEmbeddingEngine_FinalizeConfiguration_Params params;
    StatusHelper status;
    params.status = status.c_status;

    params.host_config = hosts_config.data();
    params.host_config_size = hosts_config.size();

    ResourceMgr* rm = GetTPUConfigResourceMgr();
    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);
    params.tpu_mesh_state = mesh_state->data();

    tpu::OpsApiFn()->TpuEmbeddingEngine_FinalizeConfigurationFn(&params);
    OP_REQUIRES_OK(ctx, status.status());
  }

  ~FinalizeTPUEmbeddingSystemConfigurationOp() override {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FinalizeTPUEmbeddingSystemConfigurationOp);
};

// The IsTPUEmbeddingInitializedOp is used to check whether the TPU
// TPUEmbedding Embedding has been initialized. It takes no argument and outputs
// a boolean value which indicates the TPUEmbedding Embedding is initialized or
// not. It runs on the CPU device.
class IsTPUEmbeddingInitializedOp : public OpKernel {
 public:
  explicit IsTPUEmbeddingInitializedOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "IsTPUEmbeddingInitializedOp::Compute";

    TpuEmbeddingEngine_IsInitialized_Params params;
    StatusHelper status;
    params.status = status.c_status;

    params.config_string = config_string_.c_str();
    params.config_string_size = config_string_.size();
    bool is_initialized = false;
    params.is_tpu_embedding_initialized = &is_initialized;

    tpu::OpsApiFn()->TpuEmbeddingEngine_IsInitializedFn(&params);

    OP_REQUIRES_OK(ctx, status.status());

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    output->flat<bool>()(0) = is_initialized;

    VLOG(1) << "IsTPUEmbeddingInitializedOp::Compute done";
  }
  ~IsTPUEmbeddingInitializedOp() override {}

 private:
  std::string config_string_;
  TF_DISALLOW_COPY_AND_ASSIGN(IsTPUEmbeddingInitializedOp);
};

// These ops execute on the TPU device, so that they can access
// the JF node interfaces stored in the JF device's resource manager.
REGISTER_KERNEL_BUILDER(Name("_ExecuteTPUEmbeddingPartitioner")
                            .Device(DEVICE_CPU)
                            .HostMemory("common_config"),
                        ExecuteTPUEmbeddingPartitionerOp);
REGISTER_KERNEL_BUILDER(Name("_ConfigureTPUEmbeddingMemory")
                            .Device(DEVICE_CPU)
                            .HostMemory("common_config")
                            .HostMemory("task_host_config"),
                        ConfigureTPUEmbeddingMemoryOp);
REGISTER_KERNEL_BUILDER(Name("_ConfigureTPUEmbeddingHost")
                            .Device(DEVICE_CPU)
                            .HostMemory("common_config")
                            .HostMemory("task_host_config")
                            .HostMemory("host_config"),
                        ConfigureTPUEmbeddingHostOp);
REGISTER_KERNEL_BUILDER(Name("_ConnectInterTPUEmbeddingCommunication")
                            .Device(DEVICE_CPU)
                            .HostMemory("host_config"),
                        ConnectInterTPUEmbeddingCommunicationOp);
REGISTER_KERNEL_BUILDER(Name("_FinalizeTPUEmbeddingSystemConfiguration")
                            .Device(DEVICE_CPU)
                            .HostMemory("host_config"),
                        FinalizeTPUEmbeddingSystemConfigurationOp);
REGISTER_KERNEL_BUILDER(Name("IsTPUEmbeddingInitialized").Device(DEVICE_CPU),
                        IsTPUEmbeddingInitializedOp);

}  // namespace tensorflow

#endif  // LIBTPU_ON_GCE
