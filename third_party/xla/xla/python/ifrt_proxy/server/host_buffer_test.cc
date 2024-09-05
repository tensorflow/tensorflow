// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/server/host_buffer.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::Pointee;
using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(HostBufferStoreTest, ReadAfterWrite) {
  HostBufferStore store;
  const uint64_t kHandle = 1;

  ASSERT_THAT(store.Store(kHandle, "foo"), IsOk());
  EXPECT_THAT(store.Lookup(kHandle), IsOkAndHolds(Pointee(std::string("foo"))));

  ASSERT_THAT(store.Delete(kHandle), IsOk());
  EXPECT_THAT(store.Lookup(kHandle), StatusIs(absl::StatusCode::kNotFound));
}

TEST(HostBufferStoreTest, UnknownHandle) {
  HostBufferStore store;
  const uint64_t kHandle = 1;

  EXPECT_THAT(store.Lookup(kHandle), StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(store.Delete(kHandle), StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
