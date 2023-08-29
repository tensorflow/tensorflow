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
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_test.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_test_base.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace pjrt {
namespace {

const bool kUnused = (RegisterPjRtCApiTestFactory([]() { return GetPjrtApi(); },
                                                  /*platform_name=*/"gpu"),
                      true);

class PjrtCApiGpuTest : public PjrtCApiTestBase {
 public:
  PjrtCApiGpuTest() : PjrtCApiTestBase(GetPjrtApi()) {}
};

std::unique_ptr<::pjrt::PJRT_KeyValueCallbackData> CreateTestCKVCallback(
    absl::flat_hash_map<std::string, std::string>* kv_store, absl::Mutex& mu) {
  xla::PjRtClient::KeyValueGetCallback kv_get =
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
  xla::PjRtClient::KeyValuePutCallback kv_put =
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
    std::vector<PJRT_NamedValue>& c_options) {
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
      TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                              ::pjrt::ConvertToPjRtNamedValueList(options));
      TF_ASSERT_OK_AND_ASSIGN(
          PJRT_Client_Create_Args create_arg,
          BuildCreateArg(kv_callback_data.get(), c_options));
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
