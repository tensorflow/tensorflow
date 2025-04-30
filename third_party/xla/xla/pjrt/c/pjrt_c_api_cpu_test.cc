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

}  // namespace
}  // namespace pjrt
