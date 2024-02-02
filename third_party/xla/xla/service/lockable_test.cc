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

#include "xla/service/lockable.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/synchronization/blocking_counter.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla {

tsl::thread::ThreadPool CreateThreadPool(int32_t size) {
  return tsl::thread::ThreadPool(tsl::Env::Default(), "lockable_test", size);
}

template <>
struct LockableName<std::string> {
  static std::string ToString(const std::string& str) {
    return "lockable string " + str;
  }
};

class LockableString : public Lockable<std::string> {
  using Lockable::Lockable;
};

TEST(LockableTest, LockProperties) {
  // Lock can be default constructed and implicitly casted to bool.
  LockableString::Lock lock0;
  EXPECT_FALSE(lock0);

  // Lock can be locked from a lockable object.
  LockableString str("foo");
  LockableString::Lock lock1 = str.Acquire();
  EXPECT_TRUE(lock1);

  // Lock can be moved.
  LockableString::Lock lock2 = std::move(lock1);
  EXPECT_FALSE(lock1);
  EXPECT_TRUE(lock2);

  // TryAcquire will return empty lock for locked object.
  LockableString::Lock lock3 = str.TryAcquire();
  EXPECT_FALSE(lock3);

  // Locks have human readable names.
  EXPECT_EQ(lock1.ToString(), "<empty lock>");
  EXPECT_EQ(lock2.ToString(), "lockable string foo");

  // Lockable has human readable name.
  EXPECT_EQ(str.ToString(), "lockable string foo");

  // After lock is destructed we can acquire lockable with TryLock.
  auto sink = [](LockableString::Lock) {};
  sink(std::move(lock2));

  LockableString::Lock lock4 = str.TryAcquire();
  EXPECT_TRUE(lock4);
}

TEST(LockableTest, ExclusiveAccess) {
  absl::BlockingCounter counter(100);
  auto thread_pool = CreateThreadPool(10);

  LockableString str("foo");

  for (size_t i = 0; i < 100; ++i) {
    thread_pool.Schedule([&] {
      {  // Decrement counter only after lock is released.
        auto exclusive_str = str.Acquire();
        ASSERT_EQ(*exclusive_str, "foo");
      }
      counter.DecrementCount();
    });
  }

  counter.Wait();
}

}  // namespace xla
