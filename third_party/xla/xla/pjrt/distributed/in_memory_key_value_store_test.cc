/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/distributed/in_memory_key_value_store.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(InMemoryKeyValueStoreTest, DefaultConstructorAllowsOverwrite) {
  InMemoryKeyValueStore store;
  TF_ASSERT_OK(store.Set("key1", "val1"));
  EXPECT_THAT(store.TryGet("key1"), IsOkAndHolds("val1"));
  TF_ASSERT_OK(store.Set("key1", "val2"));
  EXPECT_THAT(store.TryGet("key1"), IsOkAndHolds("val2"));
}

TEST(InMemoryKeyValueStoreTest, ExplicitConstructorAllowOverwriteTrue) {
  InMemoryKeyValueStore store(/*allow_overwrite=*/true);
  TF_ASSERT_OK(store.Set("key1", "val1"));
  EXPECT_THAT(store.TryGet("key1"), IsOkAndHolds("val1"));
  TF_ASSERT_OK(store.Set("key1", "val2"));
  EXPECT_THAT(store.TryGet("key1"), IsOkAndHolds("val2"));
}

TEST(InMemoryKeyValueStoreTest, ExplicitConstructorAllowOverwriteFalse) {
  InMemoryKeyValueStore store(/*allow_overwrite=*/false);
  TF_ASSERT_OK(store.Set("key1", "val1"));
  EXPECT_THAT(store.Set("key1", "val2"),
              StatusIs(absl::StatusCode::kAlreadyExists));
  EXPECT_THAT(store.TryGet("key1"), IsOkAndHolds("val1"));
}

TEST(InMemoryKeyValueStoreTest, GetExistingKey) {
  InMemoryKeyValueStore store;
  TF_ASSERT_OK(store.Set("key1", "val1"));
  EXPECT_THAT(store.Get("key1", absl::Seconds(1)), IsOkAndHolds("val1"));
}

TEST(InMemoryKeyValueStoreTest, GetMissingKeyTimeout) {
  InMemoryKeyValueStore store;
  EXPECT_THAT(store.Get("missing_key", absl::Milliseconds(50)),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(InMemoryKeyValueStoreTest, GetBlocksUntilKeySet) {
  InMemoryKeyValueStore store;
  tsl::Env* env = tsl::Env::Default();
  absl::Notification started;
  std::unique_ptr<tsl::Thread> thread(
      env->StartThread(tsl::ThreadOptions(), "setter_thread", [&]() {
        started.WaitForNotification();
        absl::SleepFor(absl::Milliseconds(50));
        TF_EXPECT_OK(store.Set("key1", "val1"));
      }));

  started.Notify();
  EXPECT_THAT(store.Get("key1", absl::Seconds(5)), IsOkAndHolds("val1"));
}

TEST(InMemoryKeyValueStoreTest, TryGetExistingKey) {
  InMemoryKeyValueStore store;
  TF_ASSERT_OK(store.Set("key1", "val1"));
  EXPECT_THAT(store.TryGet("key1"), IsOkAndHolds("val1"));
}

TEST(InMemoryKeyValueStoreTest, TryGetMissingKey) {
  InMemoryKeyValueStore store;
  EXPECT_THAT(store.TryGet("missing_key"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(InMemoryKeyValueStoreTest, AsyncGetExistingKey) {
  InMemoryKeyValueStore store;
  TF_ASSERT_OK(store.Set("key1", "val1"));

  absl::StatusOr<std::string> result;
  bool called = false;
  auto call_opts =
      store.AsyncGet("key1", [&](const absl::StatusOr<std::string>& res) {
        called = true;
        result = res;
      });

  EXPECT_TRUE(called);
  EXPECT_THAT(result, IsOkAndHolds("val1"));
  EXPECT_NE(call_opts, nullptr);
}

TEST(InMemoryKeyValueStoreTest, AsyncGetBeforeSet) {
  InMemoryKeyValueStore store;

  absl::StatusOr<std::string> result;
  bool called = false;
  auto call_opts =
      store.AsyncGet("key1", [&](const absl::StatusOr<std::string>& res) {
        called = true;
        result = res;
      });

  EXPECT_FALSE(called);
  TF_ASSERT_OK(store.Set("key1", "val1"));
  EXPECT_TRUE(called);
  EXPECT_THAT(result, IsOkAndHolds("val1"));
}

TEST(InMemoryKeyValueStoreTest, AsyncGetMultipleCallbacksSameKey) {
  InMemoryKeyValueStore store;

  std::vector<absl::StatusOr<std::string>> results(3);
  std::vector<bool> called(3, false);

  for (int i = 0; i < 3; ++i) {
    store.AsyncGet(
        "key1", [i, &results, &called](const absl::StatusOr<std::string>& res) {
          called[i] = true;
          results[i] = res;
        });
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_FALSE(called[i]);
  }

  TF_ASSERT_OK(store.Set("key1", "val1"));

  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(called[i]);
    EXPECT_THAT(results[i], IsOkAndHolds("val1"));
  }
}

TEST(InMemoryKeyValueStoreTest, AsyncGetMultipleKeys) {
  InMemoryKeyValueStore store;

  absl::StatusOr<std::string> res1, res2;
  bool called1 = false, called2 = false;

  store.AsyncGet("key1", [&](const absl::StatusOr<std::string>& res) {
    called1 = true;
    res1 = res;
  });
  store.AsyncGet("key2", [&](const absl::StatusOr<std::string>& res) {
    called2 = true;
    res2 = res;
  });

  TF_ASSERT_OK(store.Set("key1", "val1"));
  EXPECT_TRUE(called1);
  EXPECT_THAT(res1, IsOkAndHolds("val1"));
  EXPECT_FALSE(called2);

  TF_ASSERT_OK(store.Set("key2", "val2"));
  EXPECT_TRUE(called2);
  EXPECT_THAT(res2, IsOkAndHolds("val2"));
}

TEST(InMemoryKeyValueStoreTest, AsyncGetCancellation) {
  InMemoryKeyValueStore store;

  absl::StatusOr<std::string> result;
  int call_count = 0;
  auto call_opts =
      store.AsyncGet("key1", [&](const absl::StatusOr<std::string>& res) {
        ++call_count;
        result = res;
      });

  EXPECT_EQ(call_count, 0);
  call_opts->StartCancel();
  EXPECT_EQ(call_count, 1);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kCancelled));

  // Setting the key afterwards should not invoke the callback again.
  TF_ASSERT_OK(store.Set("key1", "val1"));
  EXPECT_EQ(call_count, 1);
}

TEST(InMemoryKeyValueStoreTest, AsyncGetCancellationMultipleCallbacks) {
  InMemoryKeyValueStore store;

  absl::StatusOr<std::string> result1, result2;
  int call_count1 = 0, call_count2 = 0;

  auto call_opts1 =
      store.AsyncGet("key1", [&](const absl::StatusOr<std::string>& res) {
        ++call_count1;
        result1 = res;
      });
  auto call_opts2 =
      store.AsyncGet("key1", [&](const absl::StatusOr<std::string>& res) {
        ++call_count2;
        result2 = res;
      });

  call_opts1->StartCancel();
  EXPECT_EQ(call_count1, 1);
  EXPECT_THAT(result1, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(call_count2, 0);

  TF_ASSERT_OK(store.Set("key1", "val1"));
  EXPECT_EQ(call_count1, 1);
  EXPECT_EQ(call_count2, 1);
  EXPECT_THAT(result2, IsOkAndHolds("val1"));
}

TEST(InMemoryKeyValueStoreTest, AsyncGetCancelAfterCompletionIsNoop) {
  InMemoryKeyValueStore store;
  TF_ASSERT_OK(store.Set("key1", "val1"));

  int call_count = 0;
  auto call_opts =
      store.AsyncGet("key1", [&](const absl::StatusOr<std::string>& res) {
        ++call_count;
        EXPECT_THAT(res, IsOkAndHolds("val1"));
      });

  EXPECT_EQ(call_count, 1);
  call_opts->StartCancel();
  EXPECT_EQ(call_count, 1);
}

TEST(InMemoryKeyValueStoreTest, AsyncGetCancelledOnStoreDestruction) {
  absl::StatusOr<std::string> result;
  bool called = false;
  {
    InMemoryKeyValueStore store;
    store.AsyncGet("key1", [&](const absl::StatusOr<std::string>& res) {
      called = true;
      result = res;
    });
    EXPECT_FALSE(called);
  }
  EXPECT_TRUE(called);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kCancelled));
}

TEST(InMemoryKeyValueStoreTest, ConcurrentOperations) {
  InMemoryKeyValueStore store;
  tsl::Env* env = tsl::Env::Default();

  constexpr int kNumKeys = 50;
  std::vector<std::string> results(kNumKeys);
  std::vector<absl::Notification> done_notifications(kNumKeys);
  std::vector<std::unique_ptr<tsl::Thread>> threads;

  // Register AsyncGets concurrently.
  for (int i = 0; i < kNumKeys; ++i) {
    threads.push_back(std::unique_ptr<tsl::Thread>(env->StartThread(
        tsl::ThreadOptions(), absl::StrCat("getter_", i),
        [&store, &results, &done_notifications, i]() {
          std::string key = absl::StrCat("key_", i);
          store.AsyncGet(key, [&results, &done_notifications,
                               i](const absl::StatusOr<std::string>& res) {
            if (res.ok()) {
              results[i] = *res;
            }
            done_notifications[i].Notify();
          });
        })));
  }

  // Concurrently set values.
  for (int i = 0; i < kNumKeys; ++i) {
    threads.push_back(std::unique_ptr<tsl::Thread>(env->StartThread(
        tsl::ThreadOptions(), absl::StrCat("setter_", i), [&store, i]() {
          std::string key = absl::StrCat("key_", i);
          std::string val = absl::StrCat("val_", i);
          TF_EXPECT_OK(store.Set(key, val));
        })));
  }

  // Wait for all threads to finish.
  threads.clear();

  for (int i = 0; i < kNumKeys; ++i) {
    done_notifications[i].WaitForNotification();
    EXPECT_EQ(results[i], absl::StrCat("val_", i));
    EXPECT_THAT(store.TryGet(absl::StrCat("key_", i)),
                IsOkAndHolds(absl::StrCat("val_", i)));
  }
}

TEST(InMemoryKeyValueStoreTest, DeleteExistingKey) {
  InMemoryKeyValueStore store;
  TF_ASSERT_OK(store.Set("key1", "val1"));
  EXPECT_THAT(store.TryGet("key1"), IsOkAndHolds("val1"));
  TF_ASSERT_OK(store.Delete("key1"));
  EXPECT_THAT(store.TryGet("key1"), StatusIs(absl::StatusCode::kNotFound));
}

TEST(InMemoryKeyValueStoreTest, DeleteMissingKeyIsOk) {
  InMemoryKeyValueStore store;
  TF_EXPECT_OK(store.Delete("missing_key"));
}

}  // namespace
}  // namespace xla
