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

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_gpu.h"

#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace xla {
namespace pjrt {
namespace {

class PjrtCApiGpuTest : public ::testing::Test {
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

TEST_F(PjrtCApiGpuTest, ClientProcessIndex) {
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

TEST_F(PjrtCApiGpuTest, PlatformName) {
  PJRT_Client_PlatformName_Args args;
  args.client = client_;
  args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  args.priv = nullptr;
  PJRT_Error* error = api_->PJRT_Client_PlatformName(&args);
  ASSERT_EQ(error, nullptr);
  absl::string_view platform_name(args.platform_name, args.platform_name_size);
  ASSERT_EQ("gpu", platform_name);
}

TEST_F(PjrtCApiGpuTest, ApiVersion) {
  CHECK_EQ(api_->pjrt_api_version.major_version, PJRT_API_MAJOR);
  CHECK_EQ(api_->pjrt_api_version.minor_version, PJRT_API_MINOR);
}

std::unique_ptr<::pjrt::PJRT_KeyValueCallbackData> CreateTestCKVCallback(
    absl::flat_hash_map<std::string, std::string>* kv_store, absl::Mutex& mu) {
  PjRtClient::KeyValueGetCallback kv_get =
      [kv_store, &mu](const std::string& k,
                      absl::Duration timeout) -> xla::StatusOr<std::string> {
    absl::Duration wait_interval = absl::Milliseconds(10);
    int num_retry = timeout / wait_interval;
    for (int i = 0; i < num_retry; i++) {
      {
        absl::MutexLock lock(&mu);
        auto iter = kv_store->find(k);
        if (iter != kv_store->end()) {
          return iter->second;
        }
      }
      absl::SleepFor(wait_interval);
    }
    return absl::NotFoundError(
        absl::StrCat(k, " is not found in the kv store."));
  };
  PjRtClient::KeyValuePutCallback kv_put =
      [kv_store, &mu](const std::string& k,
                      const std::string& v) -> xla::Status {
    {
      absl::MutexLock lock(&mu);
      kv_store->insert(std::pair<std::string, std::string>(k, v));
    }
    return tsl::OkStatus();
  };
  return ::pjrt::ConvertToCKeyValueCallbacks(kv_get, kv_put);
}

absl::StatusOr<PJRT_Client_Create_Args> BuildCreateArg(
    ::pjrt::PJRT_KeyValueCallbackData* kv_callback_data,
    const absl::flat_hash_map<std::string, xla::PjRtValueType>& options) {
  TF_ASSIGN_OR_RETURN(std::vector<PJRT_NamedValue> c_options,
                      ::pjrt::ConvertToPjRtNamedValueList(options));
  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.create_options = c_options.data();
  args.num_options = c_options.size();
  args.kv_get_callback = kv_callback_data->c_kv_get;
  args.kv_get_user_arg = &kv_callback_data->kv_get_c_func;
  args.kv_put_callback = kv_callback_data->c_kv_put;
  args.kv_put_user_arg = &kv_callback_data->kv_put_c_func;
  args.client = nullptr;
  return args;
}

TEST(PjrtCApiGpuKVStoreTest, CreateClientWithKVCallback) {
  auto api = GetPjrtApi();
  auto kv_store_ptr =
      std::make_shared<absl::flat_hash_map<std::string, std::string>>();
  absl::Mutex mu;
  std::shared_ptr<::pjrt::PJRT_KeyValueCallbackData> kv_callback_data =
      CreateTestCKVCallback(kv_store_ptr.get(), mu);

  int num_nodes = 2;
  std::vector<std::thread> threads;
  // `num_nodes` clients will be created on the same GPU.
  for (int i = 0; i < num_nodes; i++) {
    threads.emplace_back([api, i, num_nodes,
                          kv_callback_data = kv_callback_data,
                          kv_store_ptr = kv_store_ptr] {
      absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
          {"num_nodes", static_cast<int64_t>(num_nodes)},
          {"node_id", static_cast<int64_t>(i)}};
      TF_ASSERT_OK_AND_ASSIGN(PJRT_Client_Create_Args create_arg,
                              BuildCreateArg(kv_callback_data.get(), options));
      PJRT_Error* error = api->PJRT_Client_Create(&create_arg);
      EXPECT_EQ(error, nullptr) << error->status.message();

      PJRT_Client_Devices_Args device_args;
      device_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
      device_args.priv = nullptr;
      device_args.client = create_arg.client;

      PJRT_Error* device_error = api->PJRT_Client_Devices(&device_args);
      EXPECT_EQ(device_error, nullptr);
      EXPECT_EQ(device_args.num_devices, 2);

      PJRT_Client_AddressableDevices_Args addressable_device_args;
      addressable_device_args.struct_size =
          PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
      addressable_device_args.priv = nullptr;
      addressable_device_args.client = create_arg.client;

      PJRT_Error* addressable_device_error =
          api->PJRT_Client_AddressableDevices(&addressable_device_args);
      EXPECT_EQ(addressable_device_error, nullptr);
      EXPECT_EQ(addressable_device_args.num_addressable_devices, 1);

      PJRT_Client_Destroy_Args destroy_args;
      destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
      destroy_args.priv = nullptr;
      destroy_args.client = create_arg.client;

      PJRT_Error* destroy_error = api->PJRT_Client_Destroy(&destroy_args);
      CHECK_EQ(destroy_error, nullptr);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}
}  // namespace
}  // namespace pjrt
}  // namespace xla
