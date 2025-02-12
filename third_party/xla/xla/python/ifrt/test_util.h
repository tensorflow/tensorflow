#include <optional>

#include "absl/status/statusor.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/tsl/lib/core/status_test_util.h"
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

#ifndef XLA_PYTHON_IFRT_TEST_UTIL_H_
#define XLA_PYTHON_IFRT_TEST_UTIL_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace test_util {

// Registers an IFRT client factory function. Must be called only once.
void RegisterClientFactory(
    std::function<absl::StatusOr<std::shared_ptr<Client>>()> factory);

// Returns true iff an IFRT client factory function has been registered.
bool IsClientFactoryRegistered();

// Gets a new IFRT client using the registered client factory.
absl::StatusOr<std::shared_ptr<Client>> GetClient();

// Set a default test filter if user doesn't provide one using --gtest_filter.
void SetTestFilterIfNotUserSpecified(absl::string_view custom_filter);

// Asserts the content of an Array.
// This will blocking copy the data to host buffer.
template <typename ElementT>
void AssertPerShardData(
    tsl::RCReference<Array> actual, DType expected_dtype,
    Shape expected_per_shard_shape,
    absl::Span<const absl::Span<const ElementT>> expected_per_shard_data,
    tsl::RCReference<DeviceList> expected_device_list) {
  ASSERT_EQ(actual->dtype(), expected_dtype);
  EXPECT_THAT(GetDeviceIds(actual->sharding().devices()),
              testing::ElementsAreArray(GetDeviceIds(expected_device_list)));
  TF_ASSERT_OK_AND_ASSIGN(auto actual_per_shard_arrays,
                          actual->DisassembleIntoSingleDeviceArrays(
                              ArrayCopySemantics::kAlwaysCopy));
  ASSERT_EQ(actual_per_shard_arrays.size(), expected_per_shard_data.size());
  for (int i = 0; i < actual_per_shard_arrays.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("Shard ", i));
    const tsl::RCReference<Array>& array = actual_per_shard_arrays[i];
    ASSERT_EQ(array->shape(), expected_per_shard_shape);
    std::vector<ElementT> actual_data(expected_per_shard_shape.num_elements());
    TF_ASSERT_OK(array
                     ->CopyToHostBuffer(actual_data.data(),
                                        /*byte_strides=*/std::nullopt,
                                        ArrayCopySemantics::kAlwaysCopy)
                     .Await());
    EXPECT_THAT(actual_data,
                testing::ElementsAreArray(expected_per_shard_data[i]));
  }
}

// Helper function that makes `DeviceList` containing devices at given
// indexes (not ids) within `client.devices()`.
absl::StatusOr<tsl::RCReference<DeviceList>> GetDevices(
    Client* client, absl::Span<const int> device_indices);

// Helper function that makes `DeviceList` containing devices at given
// indexes (not ids) within `client.addressable_devices()`.
absl::StatusOr<tsl::RCReference<DeviceList>> GetAddressableDevices(
    Client* client, absl::Span<const int> device_indices);

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_TEST_UTIL_H_
