#include "tensorflow/core/common_runtime/gpu/gpu_allocator_retry.h"

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/env.h"
#include <gtest/gtest.h>

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
    retry_.DeallocateRaw(
        [this](void* p) {
          mutex_lock l(mu_);
          ++memory_capacity_;
        },
        ptr);
  }

 private:
  GPUAllocatorRetry retry_;
  void* good_ptr_ = reinterpret_cast<void*>(0xdeadbeef);
  mutex mu_;
  size_t memory_capacity_ GUARDED_BY(mu_);
  int millis_to_wait_;
};

class GPUAllocatorRetryTest : public ::testing::Test {
 protected:
  GPUAllocatorRetryTest() {}

  void LaunchConsumerThreads(int num_consumers, int cap_needed) {
    consumer_count_.resize(num_consumers, 0);
    for (int i = 0; i < num_consumers; ++i) {
      consumers_.push_back(Env::Default()->StartThread(
          ThreadOptions(), "anon_thread", [this, i, cap_needed]() {
            do {
              void* ptr = nullptr;
              for (int j = 0; j < cap_needed; ++j) {
                ptr = alloc_->AllocateRaw(16, 1);
                if (ptr == nullptr) {
                  mutex_lock l(mu_);
                  has_failed_ = true;
                  return;
                }
              }
              ++consumer_count_[i];
              for (int j = 0; j < cap_needed; ++j) {
                alloc_->DeallocateRaw(ptr);
              }
            } while (!notifier_.HasBeenNotified());
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
  std::vector<Thread*> consumers_;
  std::vector<int> consumer_count_;
  Notification notifier_;
  mutex mu_;
  bool has_failed_ GUARDED_BY(mu_) = false;
  int count_ GUARDED_BY(mu_) = 0;
};

// Verifies correct retrying when memory is slightly overcommitted but
// we allow retry.
TEST_F(GPUAllocatorRetryTest, RetrySuccess) {
  // Support up to 2 allocations simultaneously, waits up to 10 msec for
  // a chance to alloc.
  alloc_.reset(new FakeAllocator(2, 10000));
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
// and retry is not allowed.
TEST_F(GPUAllocatorRetryTest, NoRetryFail) {
  // Support up to 2 allocations simultaneously, waits up to 0 msec for
  // a chance to alloc.
  alloc_.reset(new FakeAllocator(2, 0));
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
  // Support up to 2 allocations simultaneously, waits up to 10 msec for
  // a chance to alloc.
  alloc_.reset(new FakeAllocator(2, 10000));
  // Launch 3 consumers, each of whom needs 2 units at a time.  We expect
  // deadlock where 2 consumers each hold 1 unit, and timeout trying to
  // get the second.
  LaunchConsumerThreads(3, 2);
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

}  // namespace
}  // namespace tensorflow
