/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/pjrt/plugin/stablehlo_reference/client_c_pjrt.h"

#include <cstdio>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/stablehlo_reference/client_cpp_pjrt.h"

namespace mlir::stablehlo {

using xla::PjRtClient;

std::unique_ptr<PjRtClient> GetPluginPjRtClient() {
  return CreateStablehloReferencePjrtClient();
}

// Create my client
PJRT_Error* PJRT_StablehloReferenceClient_Create(
    PJRT_Client_Create_Args* args) {
  std::unique_ptr<PjRtClient> client = GetPluginPjRtClient();
  args->client = pjrt::CreateWrapperClient(std::move(client));
  printf("Creating PJRT Client from client\n");
  return nullptr;
}

PJRT_Error* PJRT_StablehloReferenceExecuteContext_Create(
    PJRT_ExecuteContext_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "ExecuteContext not supported for client execution.")};
}

PJRT_Error* PJRT_StablehloReferenceDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "Topology not supported for client compilation.")};
}

}  // namespace mlir::stablehlo

const PJRT_Api* GetPjrtApi() {
  printf("C++ Calling GetPjrtApi");
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      mlir::stablehlo::PJRT_StablehloReferenceClient_Create,
      mlir::stablehlo::PJRT_StablehloReferenceExecuteContext_Create,
      mlir::stablehlo::PJRT_StablehloReferenceDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp,
      reinterpret_cast<PJRT_Extension_Base*>(&layouts_extension),
      pjrt::PJRT_Plugin_Attributes_Xla);

  printf("stablehlo_reference client called GetPjrtApi\n");
  return &pjrt_api;
}
