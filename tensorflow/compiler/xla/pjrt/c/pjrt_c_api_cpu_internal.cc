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

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_cpu_internal.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace pjrt {
namespace cpu_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  // TODO(b/263170683): cpu_device_count should be configurable after config
  // options can be passed to PJRT_Client_Create.
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtClient> client,
      xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/4));
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_CpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{tsl::errors::Unimplemented(
      "Topology not supported for CPU compilation.")};
}

constexpr PJRT_Api pjrt_api =
    pjrt::CreatePjrtApi(pjrt::cpu_plugin::PJRT_Client_Create,
                        pjrt::cpu_plugin::PJRT_CpuDeviceTopology_Create,
                        pjrt::PJRT_Plugin_Initialize_NoOp);

const PJRT_Api* GetCpuPjrtApi() { return &pjrt_api; }

}  // namespace cpu_plugin
}  // namespace pjrt
