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
#include "tensorflow/compiler/xla/stream_executor/tpu/status_helper.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_configuration.h"

using ::tensorflow::tpu::TPUEmbeddingConfiguration;

namespace tensorflow {

namespace {
namespace se_tpu = ::stream_executor::tpu;
}

// The ExecuteTpuEmbeddingPartitioner Op is used to run the TPUEmbedding
// partitioner as well as calculate the HBM size (in bytes) required for
// TPUEmbedding operation. It takes as input a TPUEmbeddingConfiguration proto
// which describes all the embedding tables and metadata. It should be run on
// the CPU device of host:0.
// Note that the _ConfigureDistributedTPU Op must have run before this Op so
// that the TpuTopology is added to the TPUConfigResourceMgr. Subsequent
// TPUEmbedding memory and host configuration Ops (one per host) will use the
// output of this Op.
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
    VLOG(1) << "ExecuteTPUEmbeddingPartitionerOp::Compute";

    TpuEmbeddingEngine_ExecutePartitioner_Params params;
    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();

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
      LOG(WARNING) << "ExecuteTPUEmbeddingPartitioner::Compute failed"
                   << status.status().ToString();
      return;
    }

    const std::string common_config_string =
        std::string(common_config_output, common_config_output_size);
    Tensor* output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("common_config", TensorShape({}), &output));
    output->flat<tstring>()(0) = common_config_string;

    VLOG(1) << "ExecuteTPUEmbeddingPartitionerOp::Compute done";
  }

 private:
  // The embedding layer configuration for the TPUEmbedding host software.
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExecuteTPUEmbeddingPartitionerOp);
};

// Initializes the HBM memory addresses and segments on each host.
// The Op takes as input the output of the ExecuteTPUEmbeddingPartitioner Op.
// It should be run on the CPU device of each host.
class ConfigureTPUEmbeddingMemoryOp : public OpKernel {
 public:
  explicit ConfigureTPUEmbeddingMemoryOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "ConfigureTPUEmbeddingMemoryOp::Compute";

    std::string common_config_string = ctx->input(0).flat<tstring>()(0);

    TpuEmbeddingEngine_ConfigureMemory_Params params;
    params.common_config = common_config_string.c_str();
    params.common_config_size = common_config_string.size();

    char* memory_config_output = nullptr;
    auto cleanup = absl::MakeCleanup([&memory_config_output]() {
      tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(
          memory_config_output);
    });
    size_t memory_config_output_size;
    params.memory_config_size = &memory_config_output_size;
    params.memory_config = &memory_config_output;
    params.num_inputs = ctx->num_inputs();

    StatusHelper status;
    params.status = status.c_status;

    tpu::OpsApiFn()->TpuEmbeddingEngine_ConfigureMemoryFn(&params);
    OP_REQUIRES_OK(ctx, status.status());

    const std::string memory_config_string =
        std::string(memory_config_output, memory_config_output_size);

    Tensor* output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("memory_config", TensorShape({}), &output));
    output->flat<tstring>()(0) = memory_config_string;

    VLOG(1) << "ConfigureTPUEmbeddingMemoryOp::Compute done";
  }

  ~ConfigureTPUEmbeddingMemoryOp() override {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConfigureTPUEmbeddingMemoryOp);
};

// Merges the memory configurations of all hosts into one
// tpu_embedding::HbmBuffersConfig object. The memory configuration consists of
// the HBM addresses and sizes for the segments used by TPUEmbedding. The Op
// takes as input the memory configurations, i.e., the outputs of the
// ConfigureTPUEmbeddingMemory Ops on all hosts and produces an output after
// merging them. This Op should be run on the CPU device of host:0.
class CollateTPUEmbeddingMemoryOp : public OpKernel {
 public:
  explicit CollateTPUEmbeddingMemoryOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "CollateTPUEmbeddingMemoryOp::Compute";

    std::vector<std::string> memory_config_strings(ctx->num_inputs());
    std::vector<se_tpu::SerializedProto> memory_configs(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      memory_config_strings[i] = ctx->input(i).flat<tstring>()(0);
      memory_configs[i].bytes = memory_config_strings[i].c_str();
      memory_configs[i].size = memory_config_strings[i].size();
    }

    TpuEmbeddingEngine_CollateMemory_Params params;

    params.memory_configs = memory_configs.data();
    params.memory_configs_size = memory_configs.size();

    char* merged_memory_config_output = nullptr;
    auto cleanup = absl::MakeCleanup([&merged_memory_config_output]() {
      tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(
          merged_memory_config_output);
    });

    size_t merged_memory_config_output_size;
    params.merged_memory_config_size = &merged_memory_config_output_size;
    params.merged_memory_config = &merged_memory_config_output;

    StatusHelper status;
    params.status = status.c_status;

    tpu::OpsApiFn()->TpuEmbeddingEngine_CollateMemoryFn(&params);
    OP_REQUIRES_OK(ctx, status.status());

    const std::string merged_memory_config_string = std::string(
        merged_memory_config_output, merged_memory_config_output_size);

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("merged_memory_config",
                                             TensorShape({}), &output));
    output->flat<tstring>()(0) = merged_memory_config_string;

    VLOG(1) << "CollateTPUEmbeddingMemoryOp::Compute done";
  }

  ~CollateTPUEmbeddingMemoryOp() override {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollateTPUEmbeddingMemoryOp);
};

// The ConfigureTpuEmbeddingHost op is used to set up the TPUEmbedding host
// software on a given host. It takes as input a TPUEmbeddingConfiguration
// proto which describes all the embedding tables as well as the outputs of
// the ExecuteTPUEmbeddingPartitioner and CollateTPUEmbeddingMemory ops. It
// should be run on the CPU device of each task.
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

    const std::string common_config_string = ctx->input(0).flat<tstring>()(0);
    const std::string memory_config_string = ctx->input(1).flat<tstring>()(0);

    TpuEmbeddingEngine_ConfigureHost_Params params;
    params.num_inputs = ctx->num_inputs();

    params.common_config = common_config_string.c_str();
    params.common_config_size = common_config_string.size();

    params.memory_config = memory_config_string.c_str();
    params.memory_config_size = memory_config_string.size();

    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();

    char* network_config_output = nullptr;
    auto cleanup = absl::MakeCleanup([&network_config_output]() {
      tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(
          network_config_output);
    });

    size_t network_config_output_size;
    params.network_config_size = &network_config_output_size;
    params.network_config = &network_config_output;

    StatusHelper status;
    params.status = status.c_status;

    tpu::OpsApiFn()->TpuEmbeddingEngine_ConfigureHostFn(&params);
    OP_REQUIRES_OK(ctx, status.status());

    const std::string network_config_string =
        std::string(network_config_output, network_config_output_size);

    Tensor* output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("network_config", TensorShape({}), &output));
    output->flat<tstring>()(0) = network_config_string;

    VLOG(1) << "ConfigureTPUEmbeddingHostOp::Compute done";
  }

  ~ConfigureTPUEmbeddingHostOp() override {}

 private:
  // The embedding layer configuration for the TPUEmbedding host software.
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConfigureTPUEmbeddingHostOp);
};

// The ConnectTpuEmbeddingHosts op is used to set up gRPC connections
// between instances of the TPUEmbedding host software on different hosts; it
// must be run after ConfigureTpuEmbeddingHost op has been called on each host.
// It takes as input a string from each host which describes metadata about the
// TPUEmbedding configuration on that host. It should be run on the CPU device
// of each host.
class ConnectTPUEmbeddingHostsOp : public OpKernel {
 public:
  explicit ConnectTPUEmbeddingHostsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES(ctx, ctx->num_inputs() > 0,
                errors::InvalidArgument("ConnectTPUEmbeddingHostsOp must "
                                        "have at least one input"));
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "ConnectTPUEmbeddingHostsOp::Compute";

    std::vector<std::string> network_config_strings(ctx->num_inputs());
    std::vector<se_tpu::SerializedProto> network_configs(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      network_config_strings[i] = ctx->input(i).flat<tstring>()(0);
      network_configs[i].bytes = network_config_strings[i].c_str();
      network_configs[i].size = network_config_strings[i].size();
    }

    TpuEmbeddingEngine_ConnectHosts_Params params;
    params.network_configs = network_configs.data();
    params.network_configs_size = network_configs.size();

    StatusHelper status;
    params.status = status.c_status;

    tpu::OpsApiFn()->TpuEmbeddingEngine_ConnectHostsFn(&params);
    OP_REQUIRES_OK(ctx, status.status());

    VLOG(1) << "ConnectTPUEmbeddingHostsOp::Compute done";
  }

  ~ConnectTPUEmbeddingHostsOp() override {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConnectTPUEmbeddingHostsOp);
};

// The FinalizeTpuEmbeddingOp op is used to update TpuMeshCommonState and
// TpuSystemConfiguration objects with the results of the TPU embedding
// initialization. These objects are used during XLA compilation. This op should
// be run on the CPU device of each host:0.
class FinalizeTPUEmbeddingOp : public OpKernel {
 public:
  explicit FinalizeTPUEmbeddingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES(ctx, ctx->num_inputs() == 2,
                errors::InvalidArgument("FinalizeTPUEmbeddingOp must "
                                        "have exactly two inputs."));
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "FinalizeTPUEmbeddingOp::Compute";

    const std::string common_config_string = ctx->input(0).flat<tstring>()(0);
    const std::string memory_config_string = ctx->input(1).flat<tstring>()(0);

    TpuEmbeddingEngine_Finalize_Params params;

    params.common_config = common_config_string.c_str();
    params.common_config_size = common_config_string.size();

    params.memory_config = memory_config_string.c_str();
    params.memory_config_size = memory_config_string.size();

    StatusHelper status;
    params.status = status.c_status;

    ResourceMgr* rm = GetTPUConfigResourceMgr();
    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);
    params.tpu_mesh_state = mesh_state->data();

    tpu::OpsApiFn()->TpuEmbeddingEngine_FinalizeFn(&params);
    OP_REQUIRES_OK(ctx, status.status());
    VLOG(1) << "FinalizeTPUEmbeddingOp::Compute done";
  }

  ~FinalizeTPUEmbeddingOp() override {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FinalizeTPUEmbeddingOp);
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

// These ops execute on the CPU devices of TPU worker tasks.
REGISTER_KERNEL_BUILDER(Name("ExecuteTPUEmbeddingPartitioner")
                            .Device(DEVICE_CPU)
                            .HostMemory("common_config"),
                        ExecuteTPUEmbeddingPartitionerOp);
REGISTER_KERNEL_BUILDER(Name("ConfigureTPUEmbeddingMemory")
                            .Device(DEVICE_CPU)
                            .HostMemory("common_config")
                            .HostMemory("memory_config"),
                        ConfigureTPUEmbeddingMemoryOp);
REGISTER_KERNEL_BUILDER(Name("CollateTPUEmbeddingMemory")
                            .Device(DEVICE_CPU)
                            .HostMemory("memory_configs")
                            .HostMemory("merged_memory_config"),
                        CollateTPUEmbeddingMemoryOp);
REGISTER_KERNEL_BUILDER(Name("ConfigureTPUEmbeddingHost")
                            .Device(DEVICE_CPU)
                            .HostMemory("common_config")
                            .HostMemory("memory_config")
                            .HostMemory("network_config"),
                        ConfigureTPUEmbeddingHostOp);
REGISTER_KERNEL_BUILDER(Name("ConnectTPUEmbeddingHosts")
                            .Device(DEVICE_CPU)
                            .HostMemory("network_configs"),
                        ConnectTPUEmbeddingHostsOp);
REGISTER_KERNEL_BUILDER(Name("FinalizeTPUEmbedding")
                            .Device(DEVICE_CPU)
                            .HostMemory("common_config")
                            .HostMemory("memory_config"),
                        FinalizeTPUEmbeddingOp);
REGISTER_KERNEL_BUILDER(Name("IsTPUEmbeddingInitialized").Device(DEVICE_CPU),
                        IsTPUEmbeddingInitializedOp);

}  // namespace tensorflow

#endif  // LIBTPU_ON_GCE
