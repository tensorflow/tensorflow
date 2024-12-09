/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

class FakeAllocator {
 public:
  FakeAllocator(size_t cap, int millis_to_wait)
      : memory_capacity_(cap), millis_to_wait_(millis_to_wait) {}

  // Allocate just keeps track of the number of outstanding allocations,
  // not their sizes.  Assume a constant size for each.
  void* AllocateRaw(size_t alignment, size_t num_bytes) {
    return retry_.AllocateRaw(
        [this](size_t a, size_t nb, bool v) {
          mutex_lock l(mu_);
          if (memory_capacity_ > 0) {
            --memory_capacity_;
            return good_ptr_;
          } else {
            return static_cast<void*>(nullptr);
          }
        },
        millis_to_wait_, alignment, num_bytes);
  }

  void DeallocateRaw(void* ptr) {
    mutex_lock l(mu_);
    ++memory_capacity_;
    retry_.NotifyDealloc();
  }

 private:
  AllocatorRetry retry_;
  void* good_ptr_ = reinterpret_cast<void*>(0xdeadbeef);
  mutex mu_;
  size_t memory_capacity_ TF_GUARDED_BY(mu_);
  int millis_to_wait_;
};

// GPUAllocatorRetry is a mechanism to deal with race conditions which
// are inevitable in the TensorFlow runtime where parallel Nodes can
// execute in any order.  Properly testing this feature would use real
// multi-threaded race conditions, but that leads to flaky tests as
// the expected outcome fails to occur with low but non-zero
// probability.  To make these tests reliable we simulate real race
// conditions by forcing parallel threads to take turns in the
// interesting part of their interaction with the allocator.  This
// class is the mechanism that imposes turn taking.
class AlternatingBarrier {
 public:
  explicit AlternatingBarrier(int num_users)
      : num_users_(num_users), next_turn_(0), done_(num_users, false) {}

  void WaitTurn(int user_index) {
    mutex_lock l(mu_);
    int wait_cycles = 0;
    // A user is allowed to proceed out of turn if it waits too long.
    while (next_turn_ != user_index && wait_cycles++ < 10) {
      cv_.wait_for(l, std::chrono::milliseconds(1));
    }
    if (next_turn_ == user_index) {
      IncrementTurn();
      cv_.notify_all();
    }
  }

  // When a user quits, stop reserving it a turn.
  void Done(int user_index) {
    mutex_lock l(mu_);
    done_[user_index] = true;
    if (next_turn_ == user_index) {
      IncrementTurn();
      cv_.notify_all();
    }
  }

 private:
  void IncrementTurn() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    int skipped = 0;
    while (skipped < num_users_) {
      next_turn_ = (next_turn_ + 1) % num_users_;
      if (!done_[next_turn_]) return;
      ++skipped;
    }
  }

  mutex mu_;
  condition_variable cv_;
  int num_users_;
  int next_turn_ TF_GUARDED_BY(mu_);
  std::vector<bool> done_ TF_GUARDED_BY(mu_);
};

class GPUAllocatorRetryTest : public ::testing::Test {
 protected:
  GPUAllocatorRetryTest() {}

  void LaunchConsumerThreads(int num_consumers, int cap_needed) {
    barrier_ = std::make_unique<AlternatingBarrier>(num_consumers);
    consumer_count_.resize(num_consumers, 0);
    for (int i = 0; i < num_consumers; ++i) {
      consumers_.push_back(Env::Default()->StartThread(
          ThreadOptions(), "anon_thread", [this, i, cap_needed]() {
            do {
              void* ptr = nullptr;
              for (int j = 0; j < cap_needed; ++j) {
                barrier_->WaitTurn(i);
                ptr = alloc_->AllocateRaw(16, 1);
                if (ptr == nullptr) {
                  mutex_lock l(mu_);
                  has_failed_ = true;
                  barrier_->Done(i);
                  return;
                }
              }
              ++consumer_count_[i];
              for (int j = 0; j < cap_needed; ++j) {
                barrier_->WaitTurn(i);
                alloc_->DeallocateRaw(ptr);
              }
            } while (!notifier_.HasBeenNotified());
            barrier_->Done(i);
          }));
    }
  }

  // Wait up to wait_micros microseconds for has_failed_ to equal expected,
  // then terminate all threads.
  void JoinConsumerThreads(bool expected, int wait_micros) {
    while (wait_micros > 0) {
      {
        mutex_lock l(mu_);
        if (has_failed_ == expected) break;
      }
      int interval_micros = std::min(1000, wait_micros);
      Env::Default()->SleepForMicroseconds(interval_micros);
      wait_micros -= interval_micros;
    }
    notifier_.Notify();
    for (auto c : consumers_) {
      // Blocks until thread terminates.
      delete c;
    }
  }

  std::unique_ptr<FakeAllocator> alloc_;
  std::unique_ptr<AlternatingBarrier> barrier_;
  std::vector<Thread*> consumers_;
  std::vector<int> consumer_count_;
  Notification notifier_;
  mutex mu_;
  bool has_failed_ TF_GUARDED_BY(mu_) = false;
  int count_ TF_GUARDED_BY(mu_) = 0;
};

// Verifies correct retrying when memory is slightly overcommitted but
// we allow retry.
TEST_F(GPUAllocatorRetryTest, RetrySuccess) {
  // Support up to 2 allocations simultaneously, waits up to 1000 msec for
  // a chance to alloc.
  alloc_ = std::make_unique<FakeAllocator>(2, 1000);
  // Launch 3 consumers, each of whom needs 1 unit at a time.
  LaunchConsumerThreads(3, 1);
  // This should be enough time for each consumer to be satisfied many times.
  Env::Default()->SleepForMicroseconds(50000);
  JoinConsumerThreads(false, 0);
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Consumer " << i << " is " << consumer_count_[i];
  }
  {
    mutex_lock l(mu_);
    EXPECT_FALSE(has_failed_);
  }
  EXPECT_GT(consumer_count_[0], 0);
  EXPECT_GT(consumer_count_[1], 0);
  EXPECT_GT(consumer_count_[2], 0);
}

// Verifies OutOfMemory failure when memory is slightly overcommitted
// and retry is not allowed.  Note that this test will fail, i.e. no
// memory alloc failure will be detected, if it is run in a context that
// does not permit real multi-threaded execution.
TEST_F(GPUAllocatorRetryTest, NoRetryFail) {
  // Support up to 2 allocations simultaneously, waits up to 0 msec for
  // a chance to alloc.
  alloc_ = std::make_unique<FakeAllocator>(2, 0);
  // Launch 3 consumers, each of whom needs 1 unit at a time.
  LaunchConsumerThreads(3, 1);
  Env::Default()->SleepForMicroseconds(50000);
  // Will wait up to 10 seconds for proper race condition to occur, resulting
  // in failure.
  JoinConsumerThreads(true, 10000000);
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Consumer " << i << " is " << consumer_count_[i];
  }
  {
    mutex_lock l(mu_);
    EXPECT_TRUE(has_failed_);
  }
}

// Verifies OutOfMemory failure when retry is allowed but memory capacity
// is too low even for retry.
TEST_F(GPUAllocatorRetryTest, RetryInsufficientFail) {
  // Support up to 2 allocations simultaneously, waits up to 1000 msec for
  // a chance to alloc.
  alloc_ = std::make_unique<FakeAllocator>(2, 1000);
  // Launch 3 consumers, each of whom needs 2 units at a time.  We expect
  // deadlock where 2 consumers each hold 1 unit, and timeout trying to
  // get the second.
  LaunchConsumerThreads(3, 2);
  Env::Default()->SleepForMicroseconds(50000);
  // We're forcing a race condition, so this will fail quickly, but
  // give it 10 seconds anyway.
  JoinConsumerThreads(true, 10000000);
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Consumer " << i << " is " << consumer_count_[i];
  }
  {
    mutex_lock l(mu_);
    EXPECT_TRUE(has_failed_);
  }
}

}  // namespace
}  // namespace tensorflow
