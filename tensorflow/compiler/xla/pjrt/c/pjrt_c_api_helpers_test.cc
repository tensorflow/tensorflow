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
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status.h"

namespace pjrt {
namespace {

using ::testing::HasSubstr;

TEST(PjRtCApiHelperTest, ConvertValidPjRtValueType) {
  std::vector<int64_t> int64_list = {static_cast<int64_t>(1),
                                     static_cast<int64_t>(2)};
  absl::flat_hash_map<std::string, xla::PjRtValueType> original_cpp_map = {
      {"string", "v1"},
      {"int64", static_cast<int64_t>(1)},
      {"int64_list", int64_list},
      {"float", static_cast<float>(1.0)}};

  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_map,
                          ConvertToPjRtNamedValueList(original_cpp_map));
  auto converted_back_cpp_map =
      ConvertFromPjRtNamedValueList(c_map.data(), c_map.size());

  EXPECT_THAT(converted_back_cpp_map,
              testing::UnorderedElementsAreArray(original_cpp_map));
}

TEST(PjRtCApiHelperTest, ValidOptionNameAndPjRtValueTypeIndex) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> valid_map = {
      {"string", "v1"}, {"int64", static_cast<int64_t>(1)}};

  TF_EXPECT_OK(ValidateCreateOptions(valid_map, expected));
}

TEST(PjRtCApiHelperTest, InvalidOptionName) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> invalid_map = {
      {"invalid", "v1"}};

  auto status = ValidateCreateOptions(invalid_map, expected);

  EXPECT_NE(status, tsl::OkStatus());
  EXPECT_THAT(status.message(),
              HasSubstr("Unexpected option name passed to PJRT_Client_Create"));
}

TEST(PjRtCApiHelperTest, InvalidOptionTypeIndex) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> invalid_map = {
      {"string", static_cast<int64_t>(1)}};

  auto status = ValidateCreateOptions(invalid_map, expected);

  EXPECT_NE(status, tsl::OkStatus());
  EXPECT_THAT(status.message(),
              HasSubstr("Option passed to PJRT_Client_Create with name string "
                        "has type index 1 but expected type index is 0"));
}

TEST(PjRtCApiHelperTest, Callback) {
  absl::flat_hash_map<std::string, std::string> kv_store;
  absl::Mutex mu;
  xla::PjRtClient::KeyValueGetCallback kv_get =
      [&kv_store, &mu](const std::string& k,
                       absl::Duration timeout) -> xla::StatusOr<std::string> {
    absl::Duration wait_interval = absl::Milliseconds(10);
    int num_retry = timeout / wait_interval;
    for (int i = 0; i < num_retry; i++) {
      {
        absl::MutexLock lock(&mu);
        auto iter = kv_store.find(k);
        if (iter != kv_store.end()) {
          return iter->second;
        }
      }
      absl::SleepFor(wait_interval);
    }
    return absl::NotFoundError(
        absl::StrCat(k, " is not found in the kv store."));
  };
  xla::PjRtClient::KeyValuePutCallback kv_put =
      [&kv_store, &mu](const std::string& k,
                       const std::string& v) -> xla::Status {
    {
      absl::MutexLock lock(&mu);
      kv_store[k] = v;
    }
    return tsl::OkStatus();
  };
  auto kv_callback_data = ConvertToCKeyValueCallbacks(kv_get, kv_put);
  auto converted_back_kv_get = ToCppKeyValueGetCallback(
      kv_callback_data->c_kv_get, &kv_callback_data->kv_get_c_func);
  auto converted_back_kv_put = ToCppKeyValuePutCallback(
      kv_callback_data->c_kv_put, &kv_callback_data->kv_put_c_func);

  auto s = converted_back_kv_put("key", "value");
  TF_EXPECT_OK(s);

  auto v = converted_back_kv_get("key", absl::Seconds(1));
  TF_EXPECT_OK(v.status());
  EXPECT_EQ(*v, "value");
}

}  // namespace
}  // namespace pjrt
