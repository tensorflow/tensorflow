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

#include "xla/pjrt/c/pjrt_c_api_cpu_internal.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_internal.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_memory_descriptions_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"

namespace pjrt {
namespace cpu_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  xla::CpuClientOptions options;
  options.cpu_device_count = 4;

  if (args->create_options != nullptr) {
    absl::flat_hash_map<std::string, xla::PjRtValueType> create_options =
        ConvertFromPjRtNamedValueList(args->create_options, args->num_options);
    if (create_options.contains("cpu_device_count")) {
      int64_t device_count_option =
          std::get<int64_t>(create_options["cpu_device_count"]);
      options.cpu_device_count = device_count_option;
      LOG(INFO) << "cpu_device_count set via create_options: "
                << device_count_option;
    }
  }

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetXlaPjrtCpuClient(std::move(options)));
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

PJRT_Error* PJRT_CpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{
      absl::UnimplementedError("Topology not supported for CPU compilation.")};
}

const PJRT_Api* GetCpuPjrtApi() {
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);

  static PJRT_MemoryDescriptions_Extension memory_descriptions_extension =
      pjrt::CreateMemoryDescriptionsExtension(&layouts_extension.base);

  static PJRT_FFI_Extension ffi_extension =
      pjrt::CreateFfiExtension(&memory_descriptions_extension.base);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      pjrt::cpu_plugin::PJRT_Client_Create,
      pjrt::cpu_plugin::PJRT_ExecuteContext_Create,
      pjrt::cpu_plugin::PJRT_CpuDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp, &ffi_extension.base,
      pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace cpu_plugin
}  // namespace pjrt
