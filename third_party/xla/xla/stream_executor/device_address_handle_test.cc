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
#include "xla/stream_executor/device_address_handle.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace {

TEST(DeviceAddressHandle, NullMemoryNoDeallocate) {
  DeviceAddressBase null_address;
  MockStreamExecutor executor;
  EXPECT_CALL(executor, Deallocate).Times(0);
  {
    DeviceAddressHandle releaser(&executor, null_address);
  }
}

TEST(DeviceAddressHandle, Deallocates) {
  MockStreamExecutor executor;
  DeviceAddressBase memory(&executor, sizeof(executor));
  EXPECT_CALL(executor, Deallocate).Times(1);
  {
    DeviceAddressHandle releaser(&executor, memory);
  }
}

TEST(DeviceAddressHandle, MoveDeallocatesOnce) {
  MockStreamExecutor executor;
  DeviceAddressBase memory(&executor, sizeof(executor));
  EXPECT_CALL(executor, Deallocate).Times(1);
  {
    DeviceAddressHandle releaser(&executor, memory);
    DeviceAddressHandle releaser_moved(std::move(releaser));
  }
}

TEST(DeviceAddressHandle, MoveAssignmentDeallocatesOnce) {
  MockStreamExecutor executor;
  DeviceAddressBase memory(&executor, sizeof(executor));
  EXPECT_CALL(executor, Deallocate).Times(1);
  {
    DeviceAddressHandle releaser(&executor, memory);
    DeviceAddressHandle releaser2;
    releaser2 = std::move(releaser);
  }
}

}  // namespace
}  // namespace stream_executor
