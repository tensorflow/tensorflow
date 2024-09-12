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

#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/time/time.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/stream_executor/tpu/tpu_platform.h"
#include "xla/stream_executor/tpu/tpu_topology.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_local_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_configuration_ops.h"
#include "tensorflow/core/tpu/kernels/tpu_embedding_engine_state_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_pod_state.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tpu_system_interface.h"

// Timeout for waiting for TPU devices to appear.
const absl::Duration dtensor_tpu_init_retry_timeout = absl::Seconds(30);

namespace tensorflow {
namespace dtensor {

// Attempt to delete resource_name from resource_manager's default_container.
// Returns OK if the deletion succeeded, or if the resource was not found. Else
// return the deletion error.
template <class ResourceT>
Status DeleteIfExists(ResourceMgr* resource_manager,
                      const char* resource_name) {
  VLOG(1) << "Removing resource " << resource_name << " if it exists";
  Status status = resource_manager->Delete<ResourceT>(
      resource_manager->default_container(), resource_name);
  if (status.ok()) {
    VLOG(1) << "Removed existing resource " << resource_name;
    return absl::OkStatus();
  }
  if (status.code() == error::NOT_FOUND) {
    VLOG(1) << "No resource " << resource_name << " to remove";
    return absl::OkStatus();
  }
  VLOG(1) << "Error removing resource " << resource_name << " : " << status;
  return status;
}

class ConfigureAndInitializeGlobalTPUOpKernel : public OpKernel {
 public:
  explicit ConfigureAndInitializeGlobalTPUOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_tfrt_host_runtime", &use_tfrt_host_runtime_));
  }
  void Compute(OpKernelContext* ctx) override {
    LOG(INFO) << "ConfigureAndInitializeGlobalTPUOpKernel op";

    ResourceMgr* rmgr = GetTPUConfigResourceMgr();
    std::vector<int32> core_id_output_vec;
    auto retry_timeout = dtensor_tpu_init_retry_timeout;

    TpuSystemInterface* tpu_system = GetPreferredTpuSystem();
    if (tpu_system == nullptr) {
      VLOG(1) << "Initializing the default TPU system.";
      OP_REQUIRES_OK(ctx, InitializeInternal(ctx, rmgr, retry_timeout,
                                             &core_id_output_vec));
    } else {
      VLOG(1) << "Initializing a preferred TPU system.";
      OP_REQUIRES_OK(ctx,
                     tpu_system->Initialize(
                         ctx, rmgr, retry_timeout, &core_id_output_vec,
                         /*use_tfrt_host_runtime=*/use_tfrt_host_runtime_));
    }

    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "core_id_output_vec";
      for (auto i : core_id_output_vec) {
        LOG(INFO) << i;
      }
    }

    // Set output using local core ID vector.
    Tensor* ctx_output;
    auto core_id_output_vec_size = core_id_output_vec.size();
    OP_REQUIRES_OK(
        ctx,
        ctx->allocate_output(
            0, TensorShape({static_cast<long long>(core_id_output_vec_size)}),
            &ctx_output));
    for (size_t i = 0; i < core_id_output_vec_size; ++i) {
      ctx_output->flat<int32>()(i) = core_id_output_vec[i];
    }

    LOG(INFO) << "ConfigureAndInitializeGlobalTPUOpKernel done";
  }

  ~ConfigureAndInitializeGlobalTPUOpKernel() override {}

 private:
  // ConfigureAndInitializeGlobalTPUOpKernel is neither copyable nor movable.
  ConfigureAndInitializeGlobalTPUOpKernel(
      const ConfigureAndInitializeGlobalTPUOpKernel&) = delete;
  ConfigureAndInitializeGlobalTPUOpKernel& operator=(
      const ConfigureAndInitializeGlobalTPUOpKernel&) = delete;

  bool use_tfrt_host_runtime_;

  static Status InitializeInternal(OpKernelContext* ctx, ResourceMgr* rmgr,
                                   absl::Duration retry_timeout,
                                   std::vector<int32>* core_id_output_vec) {
    // Reset the TPU embedding engine interface if we are not the master.
    // We need to reset the interface before initializing the host because the
    // resetting process reset the TPU platform.
    TF_RETURN_IF_ERROR(DeleteIfExists<tpu::TpuEmbeddingEngineStateInterface>(
        rmgr, tpu::kTpuEmbeddingEngineStateInterfaceResourceName));

    // Create the subgraph compilation cache and put it in the local resource
    // manager.
    tpu::TpuCompilationCacheInterface* compilation_cache;
    TF_RETURN_IF_ERROR(CreateTpuCompilationCache(rmgr, &compilation_cache));
    core::ScopedUnref compilation_cache_ref(compilation_cache);

    // Initialize global tpu and set `TPUHostConfiguration` with TPU topology.
    auto* tpu_platform = tpu::TpuPlatformInterface::GetRegisteredPlatform();
    if (tpu_platform == nullptr) {
      return errors::Internal("Could not find registered TPU system.");
    }

    auto start = absl::Now();
    auto init_status = absl::OkStatus();

    // Keep trying to initialize underlying TPU system until either TPU system
    // is initialized or initialization times out.
    while (!tpu_platform->Initialized() &&
           (absl::Now() - start < retry_timeout)) {
      VLOG(1) << "Initializaing global TPU system.";
      init_status = tpu_platform->Initialize();
    }
    if (!tpu_platform->Initialized()) {
      return errors::Unavailable("Unable to initialize TPU system.");
    }

    std::string host_config_serialized;
    std::vector<int32> num_device_per_host;
    const auto tpu_topology = tpu_platform->topology();
    num_device_per_host.reserve(tpu_topology.HostCount());
    for (int i = 0; i < tpu_topology.HostCount(); ++i) {
      num_device_per_host.emplace_back(tpu_topology.ChipsPerHost());
    }

    TF_RETURN_IF_ERROR(tensorflow::ConstructTpuPodState(
        rmgr, num_device_per_host, compilation_cache, &host_config_serialized));

    // Turn `host_config_serialized` into `core_id_output_vec` by calling the
    // guts of InitializeHostForDistributedTpuOp.
    TF_Status* status = TF_NewStatus();
    size_t device_id_output_size;
    int32_t* device_id_output = nullptr;
    auto cleanup = absl::MakeCleanup([&status, &device_id_output]() {
      TF_DeleteStatus(status);
      stream_executor::tpu::OpsApiFn()->TpuConfigurationApi_FreeInt32ArrayFn(
          device_id_output);
    });

    InitializeHostForDistributedTpuOp_DoWork_Params params;
    params.struct_size = InitializeHostForDistributedTpuOp_DoWork_Params_SIZE;
    params.priv = nullptr;
    params.tpu_host_config_size = host_config_serialized.size();
    params.tpu_host_config = host_config_serialized.data();
    params.enable_whole_mesh_compilations = false;
    params.is_master_worker = true;
    params.core_id_output_size = &device_id_output_size;
    params.core_id_output = &device_id_output;
    params.status = status;

    stream_executor::tpu::OpsApiFn()
        ->InitializeHostForDistributedTpuOp_DoWorkFn(&params);
    TF_RETURN_IF_ERROR(StatusFromTF_Status(status));
    for (size_t i = 0; i < device_id_output_size; ++i) {
      core_id_output_vec->push_back(device_id_output[i]);
    }

    // Create resource containers used for storing TPU topology and HBM buffer
    // configurations.
    auto delete_status = rmgr->Delete<tpu::TpuMeshStateInterface>(
        rmgr->default_container(), tpu::kTpuMeshStateInterfaceResourceName);
    if (!delete_status.ok() && delete_status.code() != error::NOT_FOUND) {
      return errors::InvalidArgument(
          "Failed to delete mesh interface. Please try initializing "
          "again once all TPU devices are allocated.");
    }

    auto* tpu_mesh = tpu::TpuMeshStateInterface::Create();
    TF_RETURN_IF_ERROR(rmgr->Create(rmgr->default_container(),
                                    tpu::kTpuMeshStateInterfaceResourceName,
                                    tpu_mesh));

    VLOG(1) << "Removing existing proto compilation cache lookup if it exists";
    Status resource_delete_status =
        rmgr->Delete<tpu::TpuCompilationCacheLookup>(
            rmgr->default_container(), tpu::kCompiledProtoCacheResourceName);

    tpu::TpuCompilationCacheInterface* local_compilation_cache;
    TF_RETURN_IF_ERROR(rmgr->Lookup(rmgr->default_container(),
                                    tpu::kCompilationCacheResourceName,
                                    &local_compilation_cache));
    local_compilation_cache->Unref();

    VLOG(1) << "Creating compilation proto cache resource";
    tpu::TpuCompilationCacheLookup* proto_lookup;
    proto_lookup =
        new tpu::TpuCompilationCacheLocalLookup(local_compilation_cache);
    TF_RETURN_IF_ERROR(rmgr->Create(rmgr->default_container(),
                                    tpu::kCompiledProtoCacheResourceName,
                                    proto_lookup));
    TF_RETURN_IF_ERROR(
        rmgr->Create(rmgr->default_container(),
                     tpu::kTpuEmbeddingEngineStateInterfaceResourceName,
                     tpu::TpuEmbeddingEngineStateInterface::Create()));

    return absl::OkStatus();
  }
};

class ShutdownTPUSystemOpKernel : public OpKernel {
 public:
  explicit ShutdownTPUSystemOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    LOG(INFO) << "ShutdownTPUSystemOpKernel op";

    Status status;
    TpuSystemInterface* tpu_system = GetPreferredTpuSystem();
    if (tpu_system == nullptr) {
      VLOG(1) << "Shutting down the default TPU system.";
      // In current runtime, we reset the TPU platform, which in turn shuts
      // down the tpu::System.
      auto* tpu_platform = tpu::TpuPlatformInterface::GetRegisteredPlatform();
      OP_REQUIRES(ctx, tpu_platform != nullptr,
                  errors::Internal("Could not find registered TPU system."));

      status = tpu_platform->Reset(/*only_tear_down=*/true,
                                   /*reason=*/"ShutdownSystem");
    } else {
      VLOG(1) << "Shutting down a preferred TPU system.";
      status = tpu_system->Shutdown();
    }

    Tensor* output_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({1}), &output_tensor));

    if (status.ok()) {
      output_tensor->flat<bool>()(0) = true;
    } else {
      output_tensor->flat<bool>()(0) = false;
    }
  }
};

class SetGlobalTPUArrayOpKernel : public OpKernel {
 public:
  explicit SetGlobalTPUArrayOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "SetGlobalTPUArrayOpKernel op";
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(ctx->input(0).shape()),
        errors::InvalidArgument("Expected argument 0 to be a scalar. Received",
                                ctx->input(0).DebugString()));
    auto tpu_topology = ctx->input(0).scalar<tstring>()();

    StatusHelper status;
    stream_executor::tpu::OpsApiFn()->SetGlobalTPUArrayOp_DoWorkFn(
        tpu_topology.size(), tpu_topology.data(), status.c_status);
    OP_REQUIRES_OK(ctx, status.status());

    VLOG(1) << "SetGlobalTPUArrayOpKernel done";
  }
};

REGISTER_KERNEL_BUILDER(Name("ConfigureAndInitializeGlobalTPU")
                            .Device(DEVICE_TPU_SYSTEM)
                            .HostMemory("output"),
                        ConfigureAndInitializeGlobalTPUOpKernel);

REGISTER_KERNEL_BUILDER(Name("ShutdownTPUSystem").Device(DEVICE_TPU_SYSTEM),
                        ShutdownTPUSystemOpKernel);

REGISTER_KERNEL_BUILDER(Name("DTensorSetGlobalTPUArray")
                            .Device(DEVICE_TPU_SYSTEM)
                            .HostMemory("topology"),
                        SetGlobalTPUArrayOpKernel);

}  // namespace dtensor
}  // namespace tensorflow
