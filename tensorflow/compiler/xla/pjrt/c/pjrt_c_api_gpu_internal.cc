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
#include <utility>

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

  // TODO(b/261916900) initializing allocator_config is important as should be
  // passed through the args later.
  xla::GpuAllocatorConfig allocator_config;
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetStreamExecutorGpuClient(
                            /*asynchronous=*/true, allocator_config,
                            /*distributed_client=*/nullptr,
                            /*node_id=*/0));
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
                        pjrt::gpu_plugin::PJRT_GpuDeviceTopology_Create);

const PJRT_Api* GetGpuPjrtApi() { return &pjrt_api; }

}  // namespace gpu_plugin
}  // namespace pjrt
