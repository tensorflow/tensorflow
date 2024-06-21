/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/backends/profiler/plugin/plugin_tracer_impl.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/backends/profiler/plugin/profiler_error.h"
#include "xla/client/local_client.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_custom_partitioner_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_internal.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/c/pjrt_c_api_stream_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/custom_partition_callback.h"
#include "xla/service/compiler.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/stream_executor/device_description.h"

namespace pjrt {
namespace gpu_plugin {

#if TENSORFLOW_USE_ROCM
#define PJRT_GPU_PLUGIN_PLATFORM_NAME "ROCM"
#else
#define PJRT_GPU_PLUGIN_PLATFORM_NAME "CUDA"
#endif

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  absl::flat_hash_map<std::string, xla::PjRtValueType> create_options =
      pjrt::ConvertFromPjRtNamedValueList(args->create_options,
                                          args->num_options);
  const auto kExpectedOptionNameAndTypes =
      absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
          {"platform_name", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
          {"allocator", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
          {"memory_fraction", PJRT_NamedValue_Type::PJRT_NamedValue_kFloat},
          {"preallocate", PJRT_NamedValue_Type::PJRT_NamedValue_kBool},
          {"collective_memory_size",
           PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
          {"visible_devices", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List},
          {"node_id", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
          {"num_nodes", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
          {"enable_mock_nccl", PJRT_NamedValue_Type::PJRT_NamedValue_kBool},
      });
  PJRT_RETURN_IF_ERROR(
      ValidateCreateOptions(create_options, kExpectedOptionNameAndTypes));

  std::optional<std::string> platform_name;
  if (auto it = create_options.find("platform_name");
      it != create_options.end()) {
    platform_name.emplace(std::get<std::string>(it->second));
  }
  xla::GpuAllocatorConfig allocator_config;
  if (auto it = create_options.find("allocator"); it != create_options.end()) {
    auto allocator_name = std::get<std::string>(it->second);
    if (allocator_name == "default") {
      allocator_config.kind = xla::GpuAllocatorConfig::Kind::kDefault;
    } else if (allocator_name == "platform") {
      allocator_config.kind = xla::GpuAllocatorConfig::Kind::kPlatform;
    } else if (allocator_name == "bfc") {
      allocator_config.kind = xla::GpuAllocatorConfig::Kind::kBFC;
    } else if (allocator_name == "cuda_async") {
      allocator_config.kind = xla::GpuAllocatorConfig::Kind::kCudaAsync;
    } else {
      return new PJRT_Error{absl::UnimplementedError(absl::StrFormat(
          "Allocator %s not supported for PJRT GPU plugin. Supported allocator "
          "options are: 'default', 'platform', 'bfc' and 'cuda_async'.",
          allocator_name))};
    }
  }
  if (auto it = create_options.find("memory_fraction");
      it != create_options.end()) {
    allocator_config.memory_fraction = std::get<float>(it->second);
  }
  if (auto it = create_options.find("preallocate");
      it != create_options.end()) {
    allocator_config.preallocate = std::get<bool>(it->second);
  }
  if (auto it = create_options.find("collective_memory_size");
      it != create_options.end()) {
    allocator_config.collective_memory_size = std::get<int64_t>(it->second);
  }
  std::optional<std::set<int>> visible_devices;
  if (auto it = create_options.find("visible_devices");
      it != create_options.end()) {
    const auto& vec = std::get<std::vector<int64_t>>(it->second);
    visible_devices.emplace(vec.begin(), vec.end());
  }
  int node_id = 0;
  if (auto it = create_options.find("node_id"); it != create_options.end()) {
    node_id = std::get<int64_t>(it->second);
  }
  int num_nodes = 1;
  if (auto it = create_options.find("num_nodes"); it != create_options.end()) {
    num_nodes = std::get<int64_t>(it->second);
  }
  bool enable_mock_nccl = false;
  if (auto it = create_options.find("enable_mock_nccl");
      it != create_options.end()) {
    enable_mock_nccl = std::get<bool>(it->second);
  }

  xla::GpuClientOptions options;
  options.allocator_config = allocator_config;
  options.node_id = node_id;
  options.num_nodes = num_nodes;
  options.allowed_devices = visible_devices;
  options.platform_name = platform_name;
  options.kv_store =
      pjrt::ToCppKeyValueStore(args->kv_get_callback, args->kv_get_user_arg,
                               args->kv_put_callback, args->kv_put_user_arg);
  options.enable_mock_nccl = enable_mock_nccl;
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetStreamExecutorGpuClient(options));
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecuteContext_Create_Args",
      PJRT_ExecuteContext_Create_Args_STRUCT_SIZE, args->struct_size));
  auto execute_context = std::make_unique<xla::ExecuteContext>();
  args->context = pjrt::CreateWrapperExecuteContext(std::move(execute_context));
  return nullptr;
}

PJRT_Error* PJRT_GpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_TopologyDescription_Create_Args",
      PJRT_TopologyDescription_Create_Args_STRUCT_SIZE, args->struct_size));

  PJRT_ASSIGN_OR_RETURN(xla::LocalClient * xla_client,
                        xla::GetGpuXlaClient(/*platform_name=*/std::nullopt,
                                             /*allowed_devices=*/std::nullopt));
  stream_executor::StreamExecutor* executor =
      xla_client->backend().default_stream_executor();
  const stream_executor::DeviceDescription& description =
      executor->GetDeviceDescription();
  std::vector<int> device_ids;
  device_ids.reserve(xla_client->backend().stream_executors().size());
  for (stream_executor::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    device_ids.push_back(executor->device_ordinal());
  }
  auto gpu_target_config = xla::Compiler::TargetConfig(executor);
  // TODO(b/341334898): Create a single-host GPU topology. Will be updated for
  // multi-host support in the future.
  auto gpu_topology = std::make_shared<const xla::GpuTopology>(
      device_ids, description.name(),
      /*num_slices=*/1,
      /*num_hosts_per_slice=*/1,
      /*num_devices_per_host=*/device_ids.size());
  auto pjrt_topology =
      std::make_unique<xla::StreamExecutorGpuTopologyDescription>(
          xla::CudaId(), xla::CudaName(), std::move(gpu_topology),
          absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>{
              {"target_config",
               gpu_target_config.ToProto().SerializeAsString()}});
  args->topology = CreateWrapperDeviceTopology(std::move(pjrt_topology));
  return nullptr;
}

PLUGIN_Profiler_Api profiler_api{
    /*struct_size=*/PLUGIN_Profiler_Api_STRUCT_SIZE,
    /*priv=*/nullptr,
    /*error_destroy=*/xla::profiler::PLUGIN_Profiler_Error_Destroy,
    /*error_message=*/xla::profiler::PLUGIN_Profiler_Error_Message,
    /*error_get_code=*/xla::profiler::PLUGIN_Profiler_Error_GetCode,
    /*create=*/xla::profiler::PLUGIN_Profiler_Create,
    /*destroy=*/xla::profiler::PLUGIN_Profiler_Destroy,
    /*start=*/xla::profiler::PLUGIN_Profiler_Start,
    /*stop=*/xla::profiler::PLUGIN_Profiler_Stop,
    /*collect_data=*/xla::profiler::PLUGIN_Profiler_CollectData,
};

PJRT_Profiler_Extension profiler_extension{
    /*struct_size=*/PJRT_Profiler_Extension_STRUCT_SIZE,
    /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Profiler,
    /*next=*/nullptr,
    /*profiler_api=*/&profiler_api,
};

PJRT_Error* PJRT_Register_Custom_Partitioner(
    PJRT_Register_Custom_Partitioner_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Register_Custom_Partitioner_Args",
      PJRT_Register_Custom_Partitioner_Args_STRUCT_SIZE, args->struct_size));
  std::string name(args->name, args->name_size);
  RegisterCustomCallPartitioner(
      name, jax::CreateCApiCustomCallPartitioner(args->callbacks));
  return nullptr;
}

PJRT_Custom_Partitioner_Extension custom_partitioner{
    /*struct_size=*/PJRT_Custom_Partitioner_Extension_STRUCT_SIZE,
    /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Custom_Partitioner,
    /*next=*/reinterpret_cast<PJRT_Extension_Base*>(&profiler_extension),
    /*register_custom_partitioner=*/PJRT_Register_Custom_Partitioner,
};

PJRT_Error* PJRT_Get_Stream_For_External_Ready_Events(
    PJRT_Get_Stream_For_External_Ready_Events_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Get_Stream_For_External_Ready_Events_Args",
      PJRT_Get_Stream_For_External_Ready_Events_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      args->stream, args->device->device->GetStreamForExternalReadyEvents());
  return nullptr;
}

PJRT_Error* PJRT_Wait_Until_Buffer_Ready_On_Stream(
    PJRT_Wait_Until_Buffer_Ready_On_Stream_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Wait_Until_Buffer_Ready_On_Stream_Args",
      PJRT_Wait_Until_Buffer_Ready_On_Stream_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference,
      args->buffer->buffer->AcquireExternalReference());
  PJRT_RETURN_IF_ERROR(
      external_reference->WaitUntilBufferReadyOnStream(args->stream));
  return nullptr;
}

PJRT_Stream_Extension stream{
    /*struct_size=*/PJRT_Stream_Extension_STRUCT_SIZE,
    /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Stream,
    /*next=*/reinterpret_cast<PJRT_Extension_Base*>(&custom_partitioner),
    /*get_stream=*/PJRT_Get_Stream_For_External_Ready_Events,
    /*wait_stream=*/PJRT_Wait_Until_Buffer_Ready_On_Stream,
};

PJRT_Error* PJRT_Gpu_Register_Custom_Call(
    PJRT_Gpu_Register_Custom_Call_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Gpu_Register_Custom_Call_Args",
      PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE, args->struct_size));
  std::string function_name(args->function_name, args->function_name_size);
  switch (args->api_version) {
    case 0:
      xla::CustomCallTargetRegistry::Global()->Register(
          function_name, args->custom_call_function,
          PJRT_GPU_PLUGIN_PLATFORM_NAME);
      return nullptr;
    case 1:
      xla::ffi::Ffi::RegisterStaticHandler(
          xla::ffi::GetXlaFfiApi(), function_name,
          PJRT_GPU_PLUGIN_PLATFORM_NAME,
          reinterpret_cast<XLA_FFI_Handler*>(args->custom_call_function));
      return nullptr;
    default:
      return new PJRT_Error{absl::UnimplementedError(
          absl::StrFormat("API version %d not supported for PJRT GPU plugin. "
                          "Supported versions are 0 and 1.",
                          args->api_version))};
  }
}

const PJRT_Api* GetGpuPjrtApi() {
  static PJRT_Gpu_Custom_Call custom_call{
      /*struct_size=*/PJRT_Gpu_Custom_Call_STRUCT_SIZE,
      /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call,
      /*next=*/reinterpret_cast<PJRT_Extension_Base*>(&stream),
      /*custom_call=*/PJRT_Gpu_Register_Custom_Call,
  };

  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(
          reinterpret_cast<PJRT_Extension_Base*>(&custom_call));

  static PJRT_FFI_Extension ffi_extension = pjrt::CreateFfiExtension(
      reinterpret_cast<PJRT_Extension_Base*>(&layouts_extension));

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      pjrt::gpu_plugin::PJRT_Client_Create,
      pjrt::gpu_plugin::PJRT_ExecuteContext_Create,
      pjrt::gpu_plugin::PJRT_GpuDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp,
      reinterpret_cast<PJRT_Extension_Base*>(&ffi_extension),
      pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace gpu_plugin
}  // namespace pjrt
