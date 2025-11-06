/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/plugin/testing/testing_c_pjrt_internal.h"

#include <cstdio>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/testing/testing_cpp_pjrt.h"

namespace testing {

// Create C++ client. Called by the C API and is the glue between the C API
// and the C++ API.
PJRT_Error* PJRT_TestingClient_Create(PJRT_Client_Create_Args* args) {
  std::unique_ptr<xla::PjRtClient> client = CreateTestingPjrtClient();
  args->client = pjrt::CreateWrapperClient(std::move(client));
  printf("Creating PJRT Client from myplugin_pjrt.cc\n");
  return nullptr;
}

PJRT_Error* PJRT_TestingExecuteContext_Create(
    PJRT_ExecuteContext_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "ExecuteContext not supported for Testing execution.")};
}

PJRT_Error* PJRT_TestingDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{absl::UnimplementedError(
      "Topology not supported for Testing compilation.")};
}

const PJRT_Api* GetTestingPjrtApi(PJRT_Extension_Base* extension_base) {
  printf("C++ Calling GetPjrtApi");
  static PJRT_Layouts_Extension layouts_extension;
  // Static for memory storage but reassigning to avoid reusing the same
  // extension when testing with many calls to GetPjrtApi.
  layouts_extension = pjrt::CreateLayoutsExtension(extension_base);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      testing::PJRT_TestingClient_Create,
      testing::PJRT_TestingExecuteContext_Create,
      testing::PJRT_TestingDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp, &layouts_extension.base,
      pjrt::PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace testing
