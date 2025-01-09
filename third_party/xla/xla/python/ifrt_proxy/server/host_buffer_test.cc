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
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/python/ifrt/future.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::Not;
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

TEST(HostBufferStoreTest, WriteAfterReadStarted) {
  HostBufferStore store;
  const uint64_t kHandle = 1;

  auto lookup_promise =
      Future<std::shared_ptr<const std::string>>::CreatePromise();
  Future<std::shared_ptr<const std::string>> lookup_fut(lookup_promise);

  absl::Notification closure_started;
  tsl::Env::Default()->SchedClosure([&]() {
    closure_started.Notify();
    lookup_promise.Set(store.Lookup(kHandle, /*timeout=*/absl::Seconds(10)));
  });

  closure_started.WaitForNotification();
  absl::SleepFor(absl::Seconds(1));

  ASSERT_THAT(store.Store(kHandle, "foo"), IsOk());
  EXPECT_THAT(lookup_fut.Await(), IsOkAndHolds(Pointee(std::string("foo"))));
}

TEST(HostBufferStoreTest, ShutdownAfterReadStarted) {
  HostBufferStore store;
  const uint64_t kHandle = 1;

  auto lookup_promise =
      Future<std::shared_ptr<const std::string>>::CreatePromise();
  Future<std::shared_ptr<const std::string>> lookup_fut(lookup_promise);

  absl::Notification closure_started;
  tsl::Env::Default()->SchedClosure([&]() {
    closure_started.Notify();
    lookup_promise.Set(
        store.Lookup(kHandle, /*timeout=*/absl::InfiniteDuration()));
  });

  closure_started.WaitForNotification();
  absl::SleepFor(absl::Seconds(1));

  store.Shutdown("test");
  EXPECT_THAT(lookup_fut.Await(), StatusIs(Not(absl::StatusCode::kOk)));
}

TEST(HostBufferStoreTest, WriteAfterShutdown) {
  HostBufferStore store;
  const uint64_t kHandle = 1;
  store.Shutdown("test");
  EXPECT_THAT(store.Store(kHandle, "foo"),
              StatusIs(Not(absl::StatusCode::kOk)));
}

TEST(HostBufferStoreTest, LookupAfterShutdown) {
  HostBufferStore store;
  const uint64_t kHandle = 1;
  ASSERT_THAT(store.Store(kHandle, "foo"), IsOk());
  store.Shutdown("test");
  EXPECT_THAT(store.Lookup(kHandle, /*timeout=*/absl::InfiniteDuration()),
              StatusIs(Not(absl::StatusCode::kOk)));
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
