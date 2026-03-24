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

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

namespace testing {
namespace {

TEST(PjRtCApiTest, GetPjrtApi) {
  const PJRT_Api* pjrt_api = GetTestingPjrtApi(nullptr);
  ASSERT_NE(pjrt_api, nullptr);
}

TEST(PjRtCApiTest, TopologyCreate) {
  const PJRT_Api* pjrt_api = GetTestingPjrtApi(nullptr);
  PJRT_TopologyDescription_Create_Args args;
  args.struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE;
  PJRT_Error* error = pjrt_api->PJRT_TopologyDescription_Create(&args);
  ASSERT_NE(error, nullptr);
  PJRT_Error_Destroy_Args destroy_args = {PJRT_Error_Destroy_Args_STRUCT_SIZE,
                                          nullptr, error};
  pjrt_api->PJRT_Error_Destroy(&destroy_args);
}

TEST(PjRtCApiTest, ClientApi) {
  const PJRT_Api* pjrt_api = GetTestingPjrtApi(nullptr);
  ASSERT_NE(pjrt_api, nullptr);

  PJRT_Client_Create_Args create_args;
  create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create_args.create_options = nullptr;
  create_args.num_options = 0;
  create_args.kv_get_callback = nullptr;
  create_args.kv_get_user_arg = nullptr;
  create_args.kv_put_callback = nullptr;
  create_args.kv_put_user_arg = nullptr;
  PJRT_Error* error = pjrt_api->PJRT_Client_Create(&create_args);
  CHECK(error == nullptr);
  PJRT_Client* client = create_args.client;

  PJRT_Client_PlatformName_Args pname_args;
  pname_args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  pname_args.client = client;
  error = pjrt_api->PJRT_Client_PlatformName(&pname_args);
  CHECK(error == nullptr);
  EXPECT_EQ(absl::string_view(pname_args.platform_name,
                              pname_args.platform_name_size),
            "testing_pjrt_client");

  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.client = client;
  error = pjrt_api->PJRT_Client_Destroy(&destroy_args);
  CHECK(error == nullptr);
}

TEST(PjRtCApiTest, Extension) {
  struct MyExtension {
    PJRT_Extension_Base base;
    int foo;
  };
  MyExtension my_extension;
  my_extension.base.struct_size = sizeof(MyExtension);
  my_extension.base.type = PJRT_Extension_Type_Unknown;
  my_extension.base.next = nullptr;
  my_extension.foo = 42;

  const PJRT_Api* pjrt_api = GetTestingPjrtApi(&my_extension.base);
  ASSERT_NE(pjrt_api, nullptr);

  const auto* layouts_extension = pjrt::FindExtension<PJRT_Layouts_Extension>(
      pjrt_api, PJRT_Extension_Type_Layouts);
  ASSERT_NE(layouts_extension, nullptr);

  const auto* unknown_extension =
      pjrt::FindExtension<MyExtension>(pjrt_api, PJRT_Extension_Type_Unknown);
  ASSERT_NE(unknown_extension, nullptr);
  EXPECT_EQ(&unknown_extension->base, &my_extension.base);
  EXPECT_EQ(unknown_extension->foo, 42);
}

}  // namespace
}  // namespace testing
