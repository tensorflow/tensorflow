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

#include "tsl/platform/mutex.h"

#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

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

TEST(ConditionVariableTest, WaitWithPredicate) {
  constexpr int kNumThreads = 4;
  mutex mu;
  condition_variable cv;
  bool ready = false;
  int count = 0;

  // Add tasks to threads that wait on the `ready` flag.
  tsl::thread::ThreadPool pool(Env::Default(),
                               "condition_variable_test_wait_with_predicate",
                               kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    pool.Schedule([&mu, &cv, &ready, &count]() {
      mutex_lock lock(mu);
      cv.wait(lock, [&ready] { return ready; });
      ++count;
      cv.notify_one();
    });
  }

  // Verify threads are still waiting.
  {
    mutex_lock lock(mu);
    EXPECT_EQ(count, 0);
  }

  // Start worker threads.
  {
    mutex_lock lock(mu);
    ready = true;
    cv.notify_all();
  }

  // Wait for workers to complete.
  {
    mutex_lock lock(mu);
    // NOLINTNEXTLINE: MSVC requires kNumThreads to be captured.
    cv.wait(lock, [&count, kNumThreads] { return count == kNumThreads; });
    EXPECT_EQ(count, kNumThreads);
  }
}

TEST(ConditionVariableTest, WaitWithTruePredicateDoesntBlock) {
  mutex mu;
  mutex_lock lock(mu);
  condition_variable cv;

  // CV doesn't wait if predicate is true.
  cv.wait(lock, [] { return true; });

  // Verify the lock is still locked.
  EXPECT_TRUE(static_cast<bool>(lock));
}

}  // namespace
}  // namespace tsl
