/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/service/custom_call_target_registry.h"
#include "tsl/platform/errors.h"

namespace pjrt {
namespace gpu_plugin {

#define PJRT_GPU_PLUGIN_PLATFORM_NAME "CUDA"

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  absl::flat_hash_map<std::string, xla::PjRtValueType> create_options =
      pjrt::ConvertFromPjRtNamedValueList(args->create_options,
                                          args->num_options);
  const auto kExpectedOptionNameAndTypes =
      absl::flat_hash_map<std::string, PJRT_NamedValue_Type>(
          {{"allocator", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
           {"memory_fraction", PJRT_NamedValue_Type::PJRT_NamedValue_kFloat},
           {"preallocate", PJRT_NamedValue_Type::PJRT_NamedValue_kBool},
           {"visible_devices",
            PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List},
           {"node_id", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
           {"num_nodes", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64}});
  PJRT_RETURN_IF_ERROR(
      ValidateCreateOptions(create_options, kExpectedOptionNameAndTypes));

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

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetStreamExecutorGpuClient(
                            /*asynchronous=*/true, allocator_config, node_id,
                            num_nodes, visible_devices,
                            /*platform_name=*/std::nullopt, true,
                            pjrt::ToCppKeyValueGetCallback(
                                args->kv_get_callback, args->kv_get_user_arg),
                            pjrt::ToCppKeyValuePutCallback(
                                args->kv_put_callback, args->kv_put_user_arg)));
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_GpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{tsl::errors::Unimplemented(
      "Topology not supported for GPU compilation.")};
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
    /*type=*/PJRT_Structure_Type::PJRT_Structure_Type_Profiler,
    /*next=*/nullptr,
    /*profiler_api=*/&profiler_api,
};

PJRT_Error* PJRT_Gpu_Register_Custom_Call(
    PJRT_Gpu_Register_Custom_Call_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Gpu_Register_Custom_Call_Args",
      PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE, args->struct_size));
  std::string function_name(args->function_name, args->function_name_size);
  xla::CustomCallTargetRegistry::Global()->Register(
      function_name, args->custom_call_function, PJRT_GPU_PLUGIN_PLATFORM_NAME);
  return nullptr;
}

PJRT_Gpu_Custom_Call custom_call{
    /*type=*/PJRT_Structure_Type::PJRT_Structure_Type_Gpu_Custom_Call,
    /*next=*/&profiler_extension,
    /*custom_call=*/PJRT_Gpu_Register_Custom_Call,
};

constexpr PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
    pjrt::gpu_plugin::PJRT_Client_Create,
    pjrt::gpu_plugin::PJRT_GpuDeviceTopology_Create,
    pjrt::PJRT_Plugin_Initialize_NoOp, static_cast<void*>(&custom_call));

const PJRT_Api* GetGpuPjrtApi() { return &pjrt_api; }

}  // namespace gpu_plugin
}  // namespace pjrt
