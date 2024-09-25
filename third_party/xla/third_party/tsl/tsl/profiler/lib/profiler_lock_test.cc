/* Copyright 2022 The TensorFlow Authors All Rights Reserved.

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
#include "tsl/profiler/lib/profiler_lock.h"

#include <utility>

#include "absl/status/statusor.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

TEST(ProfilerLockTest, DefaultConstructorCreatesInactiveInstance) {
  ProfilerLock profiler_lock;
  EXPECT_FALSE(profiler_lock.Active());
}

TEST(ProfilerLockTest, AcquireAndReleaseExplicitly) {
  absl::StatusOr<ProfilerLock> profiler_lock = ProfilerLock::Acquire();
  ASSERT_TRUE(profiler_lock.ok());
  EXPECT_TRUE(profiler_lock->Active());
  profiler_lock->ReleaseIfActive();
  EXPECT_FALSE(profiler_lock->Active());
}

TEST(ProfilerLockTest, AcquireAndReleaseOnDestruction) {
  absl::StatusOr<ProfilerLock> profiler_lock = ProfilerLock::Acquire();
  ASSERT_TRUE(profiler_lock.ok());
  EXPECT_TRUE(profiler_lock->Active());
}

TEST(ProfilerLockTest, ReacquireWithoutReleaseFails) {
  absl::StatusOr<ProfilerLock> profiler_lock_1 = ProfilerLock::Acquire();
  absl::StatusOr<ProfilerLock> profiler_lock_2 = ProfilerLock::Acquire();
  ASSERT_TRUE(profiler_lock_1.ok());
  EXPECT_TRUE(profiler_lock_1->Active());
  EXPECT_FALSE(profiler_lock_2.ok());
}

TEST(ProfilerLockTest, ReacquireAfterReleaseSucceeds) {
  auto profiler_lock_1 = ProfilerLock::Acquire();
  ASSERT_TRUE(profiler_lock_1.ok());
  ASSERT_TRUE(profiler_lock_1->Active());
  profiler_lock_1->ReleaseIfActive();
  ASSERT_FALSE(profiler_lock_1->Active());
  auto profiler_lock_2 = ProfilerLock::Acquire();
  EXPECT_TRUE(profiler_lock_2.ok());
  EXPECT_TRUE(profiler_lock_2->Active());
}

TEST(ProfilerLockTest, InactiveAfterMove) {
  absl::StatusOr<ProfilerLock> profiler_lock_1 = ProfilerLock::Acquire();
  ASSERT_TRUE(profiler_lock_1.ok());
  ASSERT_TRUE(profiler_lock_1->Active());
  ProfilerLock profiler_lock_2 = std::move(*profiler_lock_1);
  EXPECT_FALSE(profiler_lock_1->Active());
  EXPECT_TRUE(profiler_lock_2.Active());
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
