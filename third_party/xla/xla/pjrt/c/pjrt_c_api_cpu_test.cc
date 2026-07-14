/* Copyright 2022 The OpenXLA Authors.

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
#include "xla/pjrt/c/pjrt_c_api_cpu.h"

#include <cstring>

#include <gtest/gtest.h>
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_test.h"

namespace pjrt {
namespace {

const bool kUnused = (RegisterPjRtCApiTestFactory([]() { return GetPjrtApi(); },
                                                  /*platform_name=*/"cpu"),
                      true);

TEST(PjRtCApiCpuTest, CreateClientWithCreateOptions) {
  const PJRT_Api* api_ = GetPjrtApi();

  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;

  PJRT_NamedValue create_option;
  create_option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  create_option.extension_start = nullptr;
  create_option.name = "cpu_device_count";
  create_option.name_size = strlen("cpu_device_count");
  create_option.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
  create_option.int64_value = 16;
  create_option.value_size = 1;

  args.create_options = &create_option;
  args.num_options = 1;
  args.client = nullptr;
  PJRT_Error* error = api_->PJRT_Client_Create(&args);
  ASSERT_EQ(error, nullptr);

  PJRT_Client_Devices_Args dev_args;
  dev_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
  dev_args.extension_start = nullptr;
  dev_args.client = args.client;
  error = api_->PJRT_Client_Devices(&dev_args);
  ASSERT_EQ(error, nullptr);
  ASSERT_EQ(dev_args.num_devices, 16);

  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.client = args.client;
  PJRT_Error* destroy_error = GetPjrtApi()->PJRT_Client_Destroy(&destroy_args);
  ASSERT_EQ(destroy_error, nullptr);
}

TEST(PjRtCApiCpuTest, CreateClientWithInvalidCreateOptions) {
  const PJRT_Api* api_ = GetPjrtApi();

  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;

  PJRT_NamedValue create_option;
  create_option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  create_option.extension_start = nullptr;
  create_option.name = "cpu_device_count";
  create_option.name_size = strlen("cpu_device_count");
  create_option.type = PJRT_NamedValue_Type::PJRT_NamedValue_kString;
  create_option.string_value = "invalid_int";
  create_option.value_size = strlen("invalid_int");

  args.create_options = &create_option;
  args.num_options = 1;
  args.client = nullptr;
  PJRT_Error* error = api_->PJRT_Client_Create(&args);
  ASSERT_NE(error, nullptr);

  PJRT_Error_GetCode_Args code_args;
  code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  code_args.extension_start = nullptr;
  code_args.error = error;
  ASSERT_EQ(api_->PJRT_Error_GetCode(&code_args), nullptr);
  EXPECT_EQ(code_args.code, PJRT_Error_Code_INVALID_ARGUMENT);

  PJRT_Error_Message_Args message_args;
  message_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  message_args.extension_start = nullptr;
  message_args.error = error;
  api_->PJRT_Error_Message(&message_args);
  EXPECT_NE(
      strstr(message_args.message,
             "Option passed to PJRT_Client_Create with name cpu_device_count "
             "has type index 0 but expected type index is 1"),
      nullptr);

  PJRT_Error_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.error = error;
  api_->PJRT_Error_Destroy(&destroy_args);
}

TEST(PjRtCApiCpuTest, CreateClientWithUnexpectedCreateOptions) {
  const PJRT_Api* api_ = GetPjrtApi();

  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;

  PJRT_NamedValue create_option;
  create_option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  create_option.extension_start = nullptr;
  create_option.name = "unexpected_option";
  create_option.name_size = strlen("unexpected_option");
  create_option.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
  create_option.int64_value = 16;
  create_option.value_size = 1;

  args.create_options = &create_option;
  args.num_options = 1;
  args.client = nullptr;
  PJRT_Error* error = api_->PJRT_Client_Create(&args);
  ASSERT_NE(error, nullptr);

  PJRT_Error_GetCode_Args code_args;
  code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  code_args.extension_start = nullptr;
  code_args.error = error;
  ASSERT_EQ(api_->PJRT_Error_GetCode(&code_args), nullptr);
  EXPECT_EQ(code_args.code, PJRT_Error_Code_INVALID_ARGUMENT);

  PJRT_Error_Message_Args message_args;
  message_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  message_args.extension_start = nullptr;
  message_args.error = error;
  api_->PJRT_Error_Message(&message_args);
  EXPECT_NE(strstr(message_args.message,
                   "Unexpected option name passed to PJRT_Client_Create: "
                   "unexpected_option"),
            nullptr);

  PJRT_Error_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.error = error;
  api_->PJRT_Error_Destroy(&destroy_args);
}

TEST(PjRtCApiCpuTest, CreateClientWithAsynchronousOption) {
  const PJRT_Api* api_ = GetPjrtApi();

  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;

  PJRT_NamedValue create_option;
  create_option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  create_option.extension_start = nullptr;
  create_option.name = "asynchronous";
  create_option.name_size = strlen("asynchronous");
  create_option.type = PJRT_NamedValue_Type::PJRT_NamedValue_kBool;
  create_option.bool_value = true;
  create_option.value_size = 1;

  args.create_options = &create_option;
  args.num_options = 1;
  args.client = nullptr;
  PJRT_Error* error = api_->PJRT_Client_Create(&args);
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(args.client, nullptr);

  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.client = args.client;
  ASSERT_EQ(api_->PJRT_Client_Destroy(&destroy_args), nullptr);
}

}  // namespace
}  // namespace pjrt
