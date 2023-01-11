/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_cpu.h"

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace xla {
namespace pjrt {
namespace {

class PjrtCApiCpuTest : public ::testing::Test {
 protected:
  const PJRT_Api* api_;
  PJRT_Client* client_;
  // We directly access the internal C++ client to test if the C API has the
  // same behavior as the C++ API.
  xla::PjRtClient* cc_client_;

  void SetUp() override {
    api_ = GetPjrtApi();
    client_ = make_client();
    cc_client_ = client_->client.get();
  }

  void TearDown() override { destroy_client(client_); }

  void destroy_client(PJRT_Client* client) {
    PJRT_Client_Destroy_Args destroy_args = PJRT_Client_Destroy_Args{
        .struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE,
        .priv = nullptr,
        .client = client,
    };
    PJRT_Error* error = api_->PJRT_Client_Destroy(&destroy_args);
    CHECK_EQ(error, nullptr);
  }

  PJRT_Client* make_client() {
    PJRT_Client_Create_Args create_args = PJRT_Client_Create_Args{
        .struct_size = PJRT_Client_Create_Args_STRUCT_SIZE,
        .priv = nullptr,
        .client = nullptr,
    };
    PJRT_Error* error = api_->PJRT_Client_Create(&create_args);
    CHECK_EQ(error, nullptr);
    CHECK_NE(create_args.client, nullptr);
    return create_args.client;
  }
};

TEST_F(PjrtCApiCpuTest, ClientProcessIndex) {
  PJRT_Client_ProcessIndex_Args process_index_args =
      PJRT_Client_ProcessIndex_Args{
          .struct_size = PJRT_Client_ProcessIndex_Args_STRUCT_SIZE,
          .priv = nullptr,
          .client = client_,
          .process_index = -1,
      };
  PJRT_Error* error = api_->PJRT_Client_ProcessIndex(&process_index_args);
  CHECK_EQ(error, nullptr);

  // Single-process test should return 0
  CHECK_EQ(process_index_args.process_index, 0);
}

TEST_F(PjrtCApiCpuTest, PlatformName) {
  PJRT_Client_PlatformName_Args args;
  args.client = client_;
  args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  args.priv = nullptr;
  PJRT_Error* error = api_->PJRT_Client_PlatformName(&args);
  ASSERT_EQ(error, nullptr);
  absl::string_view platform_name(args.platform_name, args.platform_name_size);
  ASSERT_EQ("cpu", platform_name);
}

}  // namespace
}  // namespace pjrt
}  // namespace xla
