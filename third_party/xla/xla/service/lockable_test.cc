/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/synchronization/blocking_counter.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

tsl::thread::ThreadPool CreateThreadPool(int32_t size) {
  return tsl::thread::ThreadPool(tsl::Env::Default(), "lockable_test", size);
}

TEST(LockableTest, ExclusiveAccess) {
  absl::BlockingCounter counter(100);
  auto thread_pool = CreateThreadPool(10);

  using LockableString = Lockable<std::string>;
  LockableString str("foo");

  for (size_t i = 0; i < 100; ++i) {
    thread_pool.Schedule([&] {
      auto exclusive_str = str.Acquire();
      ASSERT_EQ(*exclusive_str, "foo");
      counter.DecrementCount();
    });
  }

  counter.Wait();
}

}  // namespace
}  // namespace xla
