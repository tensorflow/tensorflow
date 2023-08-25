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

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "tensorflow/compiler/xla/pjrt/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"

namespace pjrt {
namespace gpu_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  absl::flat_hash_map<std::string, xla::PjRtValueType> create_options =
      pjrt::ConvertFromPjRtNamedValueList(args->create_options,
                                          args->num_options);
  const auto kExpectedOptionNameAndTypes =
      absl::flat_hash_map<std::string, PJRT_NamedValue_Type>(
          {{"visible_devices",
            PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List},
           {"node_id", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
           {"num_nodes", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64}});
  PJRT_RETURN_IF_ERROR(
      ValidateCreateOptions(create_options, kExpectedOptionNameAndTypes));

  std::optional<std::set<int>> visible_devices;
  if (auto it = create_options.find("visible_devices");
      it != create_options.end()) {
    const auto& vec = std::get<std::vector<int64_t>>(it->second);
    visible_devices->insert(vec.begin(), vec.end());
  }
  int node_id = 0;
  if (auto it = create_options.find("node_id"); it != create_options.end()) {
    node_id = std::get<int64_t>(it->second);
  }
  int num_nodes = 1;
  if (auto it = create_options.find("num_nodes"); it != create_options.end()) {
    num_nodes = std::get<int64_t>(it->second);
  }

  // TODO(b/261916900) initializing allocator_config is important as should be
  // passed through the args later.
  xla::GpuAllocatorConfig allocator_config;
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

constexpr PJRT_Api pjrt_api =
    pjrt::CreatePjrtApi(pjrt::gpu_plugin::PJRT_Client_Create,
                        pjrt::gpu_plugin::PJRT_GpuDeviceTopology_Create,
                        pjrt::PJRT_Plugin_Initialize_NoOp);

const PJRT_Api* GetGpuPjrtApi() { return &pjrt_api; }

}  // namespace gpu_plugin
}  // namespace pjrt
