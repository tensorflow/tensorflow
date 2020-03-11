/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/wait.h"

#include <atomic>
#include <condition_variable>  // NOLINT(build/c++11)
#include <mutex>               // NOLINT(build/c++11)
#include <thread>              // NOLINT(build/c++11)

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ruy/platform.h"

namespace ruy {
namespace {

// Thread taking a `value` atomic counter and incrementing it until it equals
// `end_value`, then notifying the condition variable as long as
// `value == end_value`.  If `end_value` is increased, it will then resume
// incrementing `value`, etc.  Terminates if `end_value == -1`.
class ThreadCountingUpToValue {
 public:
  ThreadCountingUpToValue(const std::atomic<int>& end_value,
                          std::atomic<int>* value,
                          std::condition_variable* condvar, std::mutex* mutex)
      : end_value_(end_value),
        value_(value),
        condvar_(condvar),
        mutex_(mutex) {}
  void operator()() {
    // end_value_==-1 is how the master thread will tell us it's OK to terminate
    while (end_value_.load() != -1) {
      // wait until end_value is set to a higher value
      while (value_->load() == end_value_.load()) {
      }
      // increment value as long as it's lower than end_value
      while (value_->fetch_add(1) < end_value_.load() - 1) {
      }
      // when value has reached end_value, notify the master thread.
      while (value_->load() == end_value_.load()) {
        std::lock_guard<std::mutex> lock(*mutex_);
        condvar_->notify_all();
      }
    }
  }

 private:
  const std::atomic<int>& end_value_;
  std::atomic<int>* value_;
  std::condition_variable* condvar_;
  std::mutex* mutex_;
};

void WaitTest(const Duration& spin_duration, const Duration& delay) {
#if RUY_PLATFORM(EMSCRIPTEN)
  // b/139927184, std::thread constructor raises exception
  return;
#endif
  std::condition_variable condvar;
  std::mutex mutex;
  std::atomic<int> value(0);
  std::atomic<int> end_value(0);
  ThreadCountingUpToValue thread_callable(end_value, &value, &condvar, &mutex);
  std::thread thread(thread_callable);
  std::this_thread::sleep_for(delay);
  for (int i = 1; i < 10; i++) {
    end_value.store(1000 * i);
    const auto& condition = [&value, &end_value]() {
      return value.load() == end_value.load();
    };
    ruy::Wait(condition, spin_duration, &condvar, &mutex);
    EXPECT_EQ(value.load(), end_value.load());
  }
  end_value.store(-1);
  thread.join();
}

TEST(WaitTest, WaitTestNoSpin) {
  WaitTest(DurationFromSeconds(0), DurationFromSeconds(0));
}

TEST(WaitTest, WaitTestSpinOneMicrosecond) {
  WaitTest(DurationFromSeconds(1e-6), DurationFromSeconds(0));
}

TEST(WaitTest, WaitTestSpinOneMillisecond) {
  WaitTest(DurationFromSeconds(1e-3), DurationFromSeconds(0));
}

TEST(WaitTest, WaitTestSpinOneSecond) {
  WaitTest(DurationFromSeconds(1), DurationFromSeconds(0));
}

// Testcase to consistently reproduce the hang in b/139062384.
TEST(WaitTest, WaitTestNoSpinWithDelayBug139062384) {
  WaitTest(DurationFromSeconds(0), DurationFromSeconds(1));
}

}  // namespace
}  // namespace ruy

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
