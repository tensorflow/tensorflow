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

#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_config_c_api.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"

namespace tensorflow {
namespace {

Status GetTpuMeshStateInterface(const ResourceMgr* rmgr,
                                tpu::TpuMeshStateInterface** state) {
  if (!rmgr->Lookup(rmgr->default_container(),
                    tpu::kTpuMeshStateInterfaceResourceName, state)
           .ok()) {
    return errors::FailedPrecondition(
        "The TPU system has not been initialized.");
  }
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

void ConfigureDistributedTpuOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "ConfigureDistributedTpuOp";
  XLA_SCOPED_LOGGING_TIMER("ConfigureDistributedTpuOp");

  std::vector<int32_t> num_devices_per_host;
  int chips_per_host = -1;
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    const Tensor& input_tensor = ctx->input(i);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(input_tensor.shape()),
        errors::InvalidArgument("Input ", i, " should be a scalar but has ",
                                input_tensor.dims(), " dimensions"));
    if (chips_per_host == -1) {
      chips_per_host = input_tensor.scalar<int32_t>()();
    } else {
      OP_REQUIRES(
          ctx, chips_per_host == input_tensor.scalar<int32>()(),
          errors::Internal("Host ", i, " has ", input_tensor.scalar<int32>()(),
                           " TPU chips but host 0 has ", chips_per_host));
    }
    num_devices_per_host.push_back(input_tensor.scalar<int32_t>()());
  }

  TF_Status* status = TF_NewStatus();
  size_t host_config_output_size;
  char* host_config_output;

  auto* rmgr = GetTPUConfigResourceMgr();
  OP_REQUIRES_OK(ctx, DeleteIfExists<tpu::TpuMeshStateInterface>(
                          rmgr, tpu::kTpuMeshStateInterfaceResourceName));

  tpu::ConfigApiFn()->ConfigureDistributedTpuOp_DoWorkFn(
      num_devices_per_host.size(), num_devices_per_host.data(),
      &host_config_output_size, &host_config_output, status);

  auto* tpu_mesh = tpu::TpuMeshStateInterface::Create();
  OP_REQUIRES_OK(
      ctx, rmgr->Create(rmgr->default_container(),
                        tpu::kTpuMeshStateInterfaceResourceName, tpu_mesh));

  Tensor* ctx_output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &ctx_output));
  ctx_output->scalar<tstring>()() =
      std::string(host_config_output, host_config_output_size);

  OP_REQUIRES_OK(ctx, StatusFromTF_Status(status));
  TF_DeleteStatus(status);

  tpu::ConfigApiFn()->TpuConfigurationApi_FreeCharArrayFn(host_config_output);

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

  TF_Status* status = TF_NewStatus();
  size_t tpu_topology_output_size;
  char* tpu_topology_output;

  tpu::TpuMeshStateInterface* mesh_state;
  auto* rmgr = GetTPUConfigResourceMgr();
  OP_REQUIRES_OK(ctx, GetTpuMeshStateInterface(rmgr, &mesh_state));
  core::ScopedUnref mesh_state_unref(mesh_state);

  auto* mesh_common_state = mesh_state->mesh_common_state();
  tpu::ConfigApiFn()->WaitForDistributedTpuOp_DoWorkFn(
      num_hosts, num_devices_per_host,
      const_cast<const int32_t**>(mapping_arg.data()), mesh_common_state,
      &tpu_topology_output_size, &tpu_topology_output, status);

  Tensor* ctx_output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &ctx_output));
  ctx_output->scalar<tstring>()() =
      std::string(tpu_topology_output, tpu_topology_output_size);

  OP_REQUIRES_OK(ctx, StatusFromTF_Status(status));
  TF_DeleteStatus(status);
  tpu::ConfigApiFn()->TpuConfigurationApi_FreeCharArrayFn(tpu_topology_output);

  VLOG(1) << "WaitForDistributedTpuOp done";
}

void ShutdownDistributedTpuOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "ShutdownDistributedTpuOp";
  XLA_SCOPED_LOGGING_TIMER("ShutdownDistributedTpuOp");

  TF_Status* status = TF_NewStatus();
  OP_REQUIRES_OK(ctx, DeleteIfExists<tpu::TpuMeshStateInterface>(
                          GetTPUConfigResourceMgr(),
                          tpu::kTpuMeshStateInterfaceResourceName));
  tpu::ConfigApiFn()->ShutdownDistributedTpuOp_DoWorkFn(status);
  OP_REQUIRES_OK(ctx, StatusFromTF_Status(status));
  TF_DeleteStatus(status);

  VLOG(1) << "ShutdownDistributedTpuOp done";
}

void InitializeHostForDistributedTpuOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "InitializeHostForDistributedTpuOp";
  XLA_SCOPED_LOGGING_TIMER("InitializeHostForDistributedTpuOp");

  auto* rmgr = GetTPUConfigResourceMgr();
  auto tpu_host_config = ctx->input(0).scalar<tstring>()();

  size_t device_id_output_size;
  int32_t* device_id_output;
  TF_Status* status = TF_NewStatus();

  bool is_master_worker =
      tpu::ConfigApiFn()->TpuConfigurationApi_HasTPUPodStateFn();
  if (!is_master_worker) {
    // Reset the mesh interface if we are not the master.
    OP_REQUIRES_OK(ctx, DeleteIfExists<tpu::TpuMeshStateInterface>(
                            rmgr, tpu::kTpuMeshStateInterfaceResourceName));
    auto* mesh_state_interface = tpu::TpuMeshStateInterface::Create();
    OP_REQUIRES_OK(ctx, rmgr->Create(rmgr->default_container(),
                                     tpu::kTpuMeshStateInterfaceResourceName,
                                     mesh_state_interface));
  }

  tpu::ConfigApiFn()->InitializeHostForDistributedTpuOp_DoWorkFn(
      tpu_host_config.size(), tpu_host_config.data(),
      enable_whole_mesh_compilations_, &device_id_output_size,
      &device_id_output, status);

  Tensor* ctx_output;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output(
               0, TensorShape({static_cast<long long>(device_id_output_size)}),
               &ctx_output));

  for (size_t i = 0; i < device_id_output_size; ++i) {
    ctx_output->flat<int32>()(i) = device_id_output[i];
  }

  OP_REQUIRES_OK(ctx, StatusFromTF_Status(status));
  TF_DeleteStatus(status);
  tpu::ConfigApiFn()->TpuConfigurationApi_FreeInt32ArrayFn(device_id_output);

  VLOG(1) << "InitializeHostForDistributedTpuOp done";
}

void SetGlobalTPUArrayOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "SetGlobalTPUArrayOp";
  XLA_SCOPED_LOGGING_TIMER("SetGlobalTPUArrayOp");

  auto tpu_topology = ctx->input(0).scalar<tstring>()();
  TF_Status* status = TF_NewStatus();

  tpu::ConfigApiFn()->SetGlobalTPUArrayOp_DoWorkFn(tpu_topology.size(),
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

  tpu::ConfigApiFn()->DisconnectDistributedTpuChipsOp_DoWorkFn(
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
