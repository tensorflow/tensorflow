/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/platform/mutex.h"

#include "tensorflow/tsl/platform/test.h"

namespace tsl {
namespace {

// Check that mutex_lock and shared_mutex_lock are movable.
class MutexTest : public ::testing::Test {
 protected:
  mutex_lock GetLock() TF_NO_THREAD_SAFETY_ANALYSIS {
    // Known false positive with thread safety analysis: the mutex is not
    // unlocked because the scoped lock doesn't destruct, which is WAI as the
    // caller will destruct it, but thread safety analysis complains.
    // See https://github.com/llvm/llvm-project/issues/58480.
    return mutex_lock{mu_};
  }

  tf_shared_lock GetSharedLock() TF_NO_THREAD_SAFETY_ANALYSIS {
    // Same known false positive as above.
    return tf_shared_lock{mu_};
  }

  // tsl::mutex does not have methods to test if a lock is held. In order to
  // test whether the lock is held, provide test-friendly wrappers around the
  // try_lock methods. These are obviously not suitable for production use, but
  // work in a single-threaded test.

  bool test_try_lock() {
    bool test = mu_.try_lock();
    if (test) mu_.unlock();
    return test;
  }

  bool test_try_lock_shared() {
    bool test = mu_.try_lock_shared();
    if (test) mu_.unlock_shared();
    return test;
  }

  mutex mu_;
};

TEST_F(MutexTest, MovableMutexLockTest) {
  // Unlocked: we can get a normal lock.
  EXPECT_TRUE(test_try_lock());
  {
    mutex_lock lock = GetLock();
    // Locked: we can't get either kind of lock.
    EXPECT_FALSE(test_try_lock());
    EXPECT_FALSE(test_try_lock_shared());
  }
  // Unlocked: we can get a normal lock.
  EXPECT_TRUE(test_try_lock());
}

TEST_F(MutexTest, SharedMutexLockTest) {
  // Unlocked: we can get a normal lock.
  EXPECT_TRUE(test_try_lock());
  {
    tf_shared_lock lock = GetSharedLock();
    // Locked in shared mode: we can't get a normal lock, but we can get a
    // shared one.
    EXPECT_FALSE(test_try_lock());
    EXPECT_TRUE(test_try_lock_shared());
  }
  // Unlocked: we can get a normal lock.
  EXPECT_TRUE(test_try_lock());
}

}  // namespace
}  // namespace tsl
