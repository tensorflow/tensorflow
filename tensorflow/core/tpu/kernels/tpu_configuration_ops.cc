/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/tpu_configuration_ops.h"

#include <cstdint>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_factory.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_local_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_embedding_engine_state_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_execute_op_options.h"
#include "tensorflow/core/tpu/kernels/tpu_fingerprint_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_pod_state.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"

namespace tensorflow {
namespace {
Status GetTpuMeshStateInterface(const ResourceMgr* rmgr,
                                tpu::TpuMeshStateInterface** state) {
  if (!rmgr->Lookup(rmgr->default_container(),
                    tpu::kTpuMeshStateInterfaceResourceName, state)
           .ok()) {
    return errors::FailedPrecondition(
        "GetTpuMeshStateInterface: The TPU system has not been initialized.");
  }
  return Status::OK();
}

Status CreateTpuFingerprintLookup(ResourceMgr* rmgr) {
  VLOG(1) << "CreateTpuFingerprintLookup";
  tpu::TpuFingerprintLookup* fingerprint_lookup;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<tpu::TpuFingerprintLookup>(
      rmgr->default_container(), tpu::kFingerprintLookupResourceName,
      &fingerprint_lookup, [&](tpu::TpuFingerprintLookup** new_lookup) {
        *new_lookup = tpu::TpuFingerprintLookup::Create();
        return Status::OK();
      }));

  core::ScopedUnref fingerprint_lookup_ref(fingerprint_lookup);
  return Status::OK();
}

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
    return Status::OK();
  }
  if (status.code() == error::NOT_FOUND) {
    VLOG(1) << "No resource " << resource_name << " to remove";
    return Status::OK();
  }
  VLOG(1) << "Error removing resource " << resource_name << " : " << status;
  return status;
}
}  // namespace

Status CreateTpuCompilationCache(
    ResourceMgr* rmgr, tpu::TpuCompilationCacheInterface** compilation_cache) {
  return rmgr->LookupOrCreate<tpu::TpuCompilationCacheInterface>(
      rmgr->default_container(), tpu::kCompilationCacheResourceName,
      compilation_cache, [&](tpu::TpuCompilationCacheInterface** new_cache) {
        *new_cache = tpu::GetCompilationCacheCreateFn()();
        return Status::OK();
      });
}

xla::StatusOr<std::vector<int32_t>> ConstructDevicesPerHost(
    OpKernelContext* ctx) {
  std::vector<int32_t> num_devices_per_host;
  int chips_per_host = -1;
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    const Tensor& input_tensor = ctx->input(i);
    if (!TensorShapeUtils::IsScalar(input_tensor.shape())) {
      return errors::InvalidArgument("Input ", i,
                                     " should be a scalar but has ",
                                     input_tensor.dims(), " dimensions");
    }
    if (chips_per_host == -1) {
      chips_per_host = input_tensor.scalar<int32_t>()();
    } else {
      if (chips_per_host != input_tensor.scalar<int32>()()) {
        return errors::Internal("Host ", i, " has ",
                                input_tensor.scalar<int32>()(),
                                " TPU chips but host 0 has ", chips_per_host);
      }
    }
    num_devices_per_host.push_back(input_tensor.scalar<int32_t>()());
  }
  return num_devices_per_host;
}

void ConfigureDistributedTpuOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "ConfigureDistributedTpuOp";
  XLA_SCOPED_LOGGING_TIMER("ConfigureDistributedTpuOp");

  xla::StatusOr<std::vector<int32_t>> num_devices_per_host =
      ConstructDevicesPerHost(ctx);
  OP_REQUIRES_OK(ctx, num_devices_per_host.status());
  ResourceMgr* rmgr = GetTPUConfigResourceMgr();

  // Create the subgraph compilation cache and put it in the local resource
  // manager.
  tpu::TpuCompilationCacheInterface* compilation_cache;
  OP_REQUIRES_OK(ctx, CreateTpuCompilationCache(rmgr, &compilation_cache));
  core::ScopedUnref compilation_cache_ref(compilation_cache);

  std::string host_config_output;
  OP_REQUIRES_OK(
      ctx, ConstructTpuPodState(rmgr, *num_devices_per_host, compilation_cache,
                                &host_config_output));

  OP_REQUIRES_OK(ctx, DeleteIfExists<tpu::TpuMeshStateInterface>(
                          rmgr, tpu::kTpuMeshStateInterfaceResourceName));

  auto* tpu_mesh = tpu::TpuMeshStateInterface::Create();
  OP_REQUIRES_OK(
      ctx, rmgr->Create(rmgr->default_container(),
                        tpu::kTpuMeshStateInterfaceResourceName, tpu_mesh));

  Tensor* ctx_output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &ctx_output));
  ctx_output->scalar<tstring>()() = std::move(host_config_output);

  OP_REQUIRES_OK(ctx, CreateTpuFingerprintLookup(rmgr));
  VLOG(1) << "ConfigureDistributedTpuOp done";
}

void WaitForDistributedTpuOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "WaitForDistributedTpuOp";
  XLA_SCOPED_LOGGING_TIMER("WaitForDistributedTpuOp");

  size_t num_devices_per_host = -1;
  size_t num_hosts = ctx->num_inputs();

  for (int i = 0; i < ctx->num_inputs(); ++i) {
    const Tensor& host_ordinal_to_global_device_id_tensor = ctx->input(i);
    OP_REQUIRES(
        ctx, host_ordinal_to_global_device_id_tensor.dims() == 1,
        errors::InvalidArgument("Input ", i, " should be a vector but has ",
                                host_ordinal_to_global_device_id_tensor.dims(),
                                " dimensions"));
  }

  std::vector<std::vector<int32_t>> mapping;
  std::vector<int32_t*> mapping_arg;

  mapping.resize(ctx->num_inputs());

  for (int i = 0; i < ctx->num_inputs(); ++i) {
    const Tensor& host_ordinal_to_global_device_id_tensor = ctx->input(i);
    const auto host_ordinal_to_global_device_id =
        host_ordinal_to_global_device_id_tensor.flat<int>();
    if (num_devices_per_host == -1) {
      num_devices_per_host =
          host_ordinal_to_global_device_id_tensor.dim_size(0);
    } else {
      OP_REQUIRES(ctx,
                  num_devices_per_host ==
                      host_ordinal_to_global_device_id_tensor.dim_size(0),
                  errors::Internal(
                      "Host ", i, " has ",
                      host_ordinal_to_global_device_id_tensor.dim_size(0),
                      " TPU devices but host 0 has ", num_devices_per_host));
    }
    for (int j = 0; j < host_ordinal_to_global_device_id_tensor.dim_size(0);
         ++j) {
      int32_t global_device_id = host_ordinal_to_global_device_id(j);
      mapping[i].push_back(global_device_id);
    }
    mapping_arg.push_back(mapping[i].data());
  }

  tpu::TpuMeshStateInterface* mesh_state;
  auto* rmgr = GetTPUConfigResourceMgr();
  OP_REQUIRES_OK(ctx, GetTpuMeshStateInterface(rmgr, &mesh_state));
  core::ScopedUnref mesh_state_unref(mesh_state);

  // TODO(b/166858751): this code to check if `TpuPodState` exists is ported
  // from a legacy library that may have staled. A candidate for cleanup.
  TpuPodState* pod_state;
  OP_REQUIRES_OK(ctx, GetTPUPodState(rmgr, &pod_state));
  core::ScopedUnref pod_state_unref(pod_state);

  size_t tpu_topology_output_size;
  char* tpu_topology_output = nullptr;
  TF_Status* status = TF_NewStatus();
  auto cleanup = absl::MakeCleanup([&status, &tpu_topology_output]() {
    TF_DeleteStatus(status);
    tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(tpu_topology_output);
  });

  auto* mesh_common_state = mesh_state->mesh_common_state();

  WaitForDistributedTpuOp_DoWork_Params params;
  params.struct_size = WaitForDistributedTpuOp_DoWork_Params_SIZE;
  params.priv = nullptr;
  params.num_hosts = num_hosts;
  params.num_cores_per_host = num_devices_per_host;
  params.host_ordinal_to_global_core_id_map =
      const_cast<const int32_t**>(mapping_arg.data());
  params.tpu_mesh_common_state = mesh_common_state;
  params.tpu_topology_output_size = &tpu_topology_output_size;
  params.tpu_topology_output = &tpu_topology_output;
  params.status = status;

  tpu::OpsApiFn()->WaitForDistributedTpuOp_DoWorkFn(&params);

  OP_REQUIRES_OK(ctx, StatusFromTF_Status(status));

  Tensor* ctx_output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &ctx_output));
  ctx_output->scalar<tstring>()() =
      std::string(tpu_topology_output, tpu_topology_output_size);

  VLOG(1) << "WaitForDistributedTpuOp done";
}

void ShutdownDistributedTpuOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "ShutdownDistributedTpuOp";
  XLA_SCOPED_LOGGING_TIMER("ShutdownDistributedTpuOp");

  auto* rmgr = GetTPUConfigResourceMgr();
  OP_REQUIRES_OK(ctx, DeleteIfExists<tpu::TpuMeshStateInterface>(
                          rmgr, tpu::kTpuMeshStateInterfaceResourceName));

  OP_REQUIRES_OK(ctx,
                 DeleteIfExists<TpuPodState>(rmgr, kTpuPodStateResourceName));
  OP_REQUIRES_OK(ctx, DeleteIfExists<tpu::TpuCompilationCacheInterface>(
                          rmgr, tpu::kCompilationCacheResourceName));

  VLOG(1) << "ShutdownDistributedTpuOp done";
}

void InitializeHostForDistributedTpuOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "InitializeHostForDistributedTpuOp";
  XLA_SCOPED_LOGGING_TIMER("InitializeHostForDistributedTpuOp");

  auto* rmgr = GetTPUConfigResourceMgr();
  auto tpu_host_config = ctx->input(0).scalar<tstring>()();

  // Reset the TPU embedding engine interface if we are not the master.
  // We need to reset the interface before initializing the host because the
  // resetting process reset the TPU platform.
  OP_REQUIRES_OK(ctx,
                 DeleteIfExists<tpu::TpuEmbeddingEngineStateInterface>(
                     rmgr, tpu::kTpuEmbeddingEngineStateInterfaceResourceName));

  bool is_master_worker =
      tpu::OpsApiFn()->TpuConfigurationApi_HasTPUPodStateFn();
  if (!is_master_worker) {
    // Reset the mesh interface if we are not the master.
    OP_REQUIRES_OK(ctx, DeleteIfExists<tpu::TpuMeshStateInterface>(
                            rmgr, tpu::kTpuMeshStateInterfaceResourceName));
    auto* mesh_state_interface = tpu::TpuMeshStateInterface::Create();
    OP_REQUIRES_OK(ctx, rmgr->Create(rmgr->default_container(),
                                     tpu::kTpuMeshStateInterfaceResourceName,
                                     mesh_state_interface));
  }

  VLOG(1) << "Removing existing proto compilation cache lookup if it exists";
  OP_REQUIRES_OK(ctx, DeleteIfExists<tpu::TpuCompilationCacheLookup>(
                          rmgr, tpu::kCompiledProtoCacheResourceName));

  if (enable_whole_mesh_compilations_) {
    // If this is a whole mesh compilation mode, create the compilation cache,
    // if missing.
    tpu::TpuCompilationCacheInterface* compilation_cache;
    OP_REQUIRES_OK(ctx, CreateTpuCompilationCache(rmgr, &compilation_cache));
    compilation_cache->Unref();
  }

  OP_REQUIRES_OK(ctx, internal::SetTpuCancellationClosesChips(
                          tpu_cancellation_closes_chips_));

  tpu::TpuCompilationCacheInterface* local_compilation_cache;
  Status s = rmgr->Lookup(rmgr->default_container(),
                          tpu::kCompilationCacheResourceName,
                          &local_compilation_cache);
  if (!s.ok()) {
    local_compilation_cache = nullptr;
  }

  TF_Status* status = TF_NewStatus();
  size_t device_id_output_size;
  int32_t* device_id_output = nullptr;
  auto cleanup = absl::MakeCleanup([&status, &device_id_output]() {
    TF_DeleteStatus(status);
    tpu::OpsApiFn()->TpuConfigurationApi_FreeInt32ArrayFn(device_id_output);
  });

  InitializeHostForDistributedTpuOp_DoWork_Params params;
  params.struct_size = InitializeHostForDistributedTpuOp_DoWork_Params_SIZE;
  params.priv = nullptr;
  params.tpu_host_config_size = tpu_host_config.size();
  params.tpu_host_config = tpu_host_config.data();
  params.enable_whole_mesh_compilations = enable_whole_mesh_compilations_;
  params.is_master_worker = is_master_worker;
  params.core_id_output_size = &device_id_output_size;
  params.core_id_output = &device_id_output;
  params.status = status;

  tpu::OpsApiFn()->InitializeHostForDistributedTpuOp_DoWorkFn(&params);
  OP_REQUIRES_OK(ctx, StatusFromTF_Status(status));

  if (local_compilation_cache != nullptr) {
    local_compilation_cache->Unref();

    tpu::TpuCompilationCacheLookup* proto_lookup;
    proto_lookup =
        new tpu::TpuCompilationCacheLocalLookup(local_compilation_cache);
    OP_REQUIRES_OK(
        ctx, rmgr->Create(rmgr->default_container(),
                          tpu::kCompiledProtoCacheResourceName, proto_lookup));
  } else {
    int64_t cache_size_bytes;
    tpu::OpsApiFn()->TpuConfigurationApi_RemoteCompilationCacheSizeInBytesFn(
        &cache_size_bytes);

    char* server_address_output = nullptr;
    auto cleanup_server_address = absl::MakeCleanup([&server_address_output]() {
      tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(
          server_address_output);
    });
    size_t server_address_output_size;

    TpuConfigurationApi_CompilationCacheServerAddrFromConfig_Params params;
    params.struct_size =
        TpuConfigurationApi_CompilationCacheServerAddrFromConfig_Params_SIZE;
    params.priv = nullptr;
    params.tpu_host_config_size = tpu_host_config.size();
    params.tpu_host_config = tpu_host_config.data();
    params.server_address_output_size = &server_address_output_size;
    params.server_address_output = &server_address_output;
    params.status = status;

    tpu::OpsApiFn()
        ->TpuConfigurationApi_CompilationCacheServerAddressFromConfigFn(
            &params);
    OP_REQUIRES_OK(ctx, StatusFromTF_Status(status));

    std::string server_address(server_address_output,
                               server_address_output_size);
    tpu::TpuCompilationCacheLookup* proto_lookup =
        new tpu::TpuCompilationCacheRpcLookup(server_address, cache_size_bytes);
    OP_REQUIRES_OK(
        ctx, rmgr->Create(rmgr->default_container(),
                          tpu::kCompiledProtoCacheResourceName, proto_lookup));
  }

  auto* engine_state_interface =
      tpu::TpuEmbeddingEngineStateInterface::Create();
  OP_REQUIRES_OK(
      ctx, rmgr->Create(rmgr->default_container(),
                        tpu::kTpuEmbeddingEngineStateInterfaceResourceName,
                        engine_state_interface));

  Tensor* ctx_output;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output(
               0, TensorShape({static_cast<long long>(device_id_output_size)}),
               &ctx_output));

  for (size_t i = 0; i < device_id_output_size; ++i) {
    ctx_output->flat<int32>()(i) = device_id_output[i];
  }
  if (ctx->function_library() != nullptr &&
      ctx->function_library()->device_mgr() != nullptr) {
    // If a DeviceMgr is available, set global IDs for TPU devices from the
    // topology.
    DeviceBase* tpu_system_device = ctx->device();
    const DeviceNameUtils::ParsedName& tpu_system_name =
        tpu_system_device->parsed_name();
    for (DeviceBase* device :
         ctx->function_library()->device_mgr()->ListDevices()) {
      const DeviceNameUtils::ParsedName& device_parsed_name =
          device->parsed_name();
      if (device_parsed_name.type == "TPU" &&
          DeviceNameUtils::IsSameAddressSpace(tpu_system_name,
                                              device_parsed_name)) {
        const DeviceBase::GpuDeviceInfo* gpu_device_info =
            device->tensorflow_gpu_device_info();
        if (gpu_device_info && gpu_device_info->stream) {
          int device_ordinal =
              gpu_device_info->stream->parent()->device_ordinal();
          if (device_ordinal >= device_id_output_size) {
            OP_REQUIRES_OK(ctx,
                           errors::Internal(absl::StrCat(
                               "TPU core with ordinal ", device_ordinal,
                               " out of range for device ", device->name(),
                               ". Expected ordinals in range [0, ",
                               device_id_output_size, ") from topology.")));
          }
          int64_t global_id = device_id_output[device_ordinal];
          VLOG(1) << "Setting global/physical id for " << device->name()
                  << " to " << global_id;
          device->set_xla_global_id(global_id);
        }
      }
    }
  }
  VLOG(1) << "InitializeHostForDistributedTpuOp done";
}

void SetGlobalTPUArrayOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "SetGlobalTPUArrayOp";
  XLA_SCOPED_LOGGING_TIMER("SetGlobalTPUArrayOp");

  auto tpu_topology = ctx->input(0).scalar<tstring>()();
  TF_Status* status = TF_NewStatus();

  tpu::OpsApiFn()->SetGlobalTPUArrayOp_DoWorkFn(tpu_topology.size(),
                                                tpu_topology.data(), status);

  OP_REQUIRES_OK(ctx, StatusFromTF_Status(status));
  TF_DeleteStatus(status);

  VLOG(1) << "SetGlobalTPUArrayOp done";
}

void DisconnectDistributedTpuChipsOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "DisconnectDistributedTpuChipsOp";
  XLA_SCOPED_LOGGING_TIMER("DisconnectDistributedTpuChipsOp");

  TF_Status* status = TF_NewStatus();
  int32_t number_of_chips_output = 0;

  tpu::OpsApiFn()->DisconnectDistributedTpuChipsOp_DoWorkFn(
      &number_of_chips_output, status);

  Tensor* ctx_output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &ctx_output));
  ctx_output->scalar<int32_t>()() = number_of_chips_output;

  OP_REQUIRES_OK(ctx, StatusFromTF_Status(status));
  TF_DeleteStatus(status);

  VLOG(1) << "DisconnectDistributedTpuChipsOp done";
}

// These ops execute on the TPU_SYSTEM device only.
REGISTER_KERNEL_BUILDER(Name("_ConfigureDistributedTPU")
                            .Device(DEVICE_TPU_SYSTEM)
                            .HostMemory("output"),
                        ConfigureDistributedTpuOp);
REGISTER_KERNEL_BUILDER(Name("_WaitForDistributedTPU")
                            .Device(DEVICE_TPU_SYSTEM)
                            .HostMemory("inputs")
                            .HostMemory("topology"),
                        WaitForDistributedTpuOp);
REGISTER_KERNEL_BUILDER(
    Name("_ShutdownDistributedTPU").Device(DEVICE_TPU_SYSTEM),
    ShutdownDistributedTpuOp);
REGISTER_KERNEL_BUILDER(Name("_InitializeHostForDistributedTPU")
                            .Device(DEVICE_TPU_SYSTEM)
                            .HostMemory("input")
                            .HostMemory("tpu_ids"),
                        InitializeHostForDistributedTpuOp);
REGISTER_KERNEL_BUILDER(
    Name("_SetGlobalTPUArray").Device(DEVICE_TPU_SYSTEM).HostMemory("topology"),
    SetGlobalTPUArrayOp);
REGISTER_KERNEL_BUILDER(Name("_DisconnectHostFromDistributedTPUSystem")
                            .Device(DEVICE_TPU_SYSTEM)
                            .HostMemory("number_of_tpu_chips"),
                        DisconnectDistributedTpuChipsOp);

}  // namespace tensorflow
