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

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/mem.h"

namespace tsl {
namespace port {

TEST(Port, AlignedMalloc) {
  for (size_t alignment = 1; alignment <= 1 << 20; alignment <<= 1) {
    void* p = AlignedMalloc(1, alignment);
    ASSERT_TRUE(p != nullptr) << "AlignedMalloc(1, " << alignment << ")";
    uintptr_t pval = reinterpret_cast<uintptr_t>(p);
    EXPECT_EQ(pval % alignment, 0);
    AlignedFree(p);
  }
}

TEST(Port, GetCurrentCPU) {
  const int cpu = GetCurrentCPU();
#if !defined(__APPLE__)
  // GetCurrentCPU does not currently work on MacOS.
  EXPECT_GE(cpu, 0);
  EXPECT_LT(cpu, NumTotalCPUs());
#endif
}

TEST(ConditionVariable, WaitForMilliseconds_Timeout) {
  absl::Mutex m;
  absl::MutexLock l(&m);
  absl::CondVar cv;
  bool result = false;
  time_t start = time(nullptr);
  // Condition variables are subject to spurious wakeups on some platforms,
  // so need to check for a timeout within a loop.
  while (!result) {
    result = cv.WaitWithTimeout(&m, absl::Milliseconds(3000));
  }
  time_t finish = time(nullptr);
  EXPECT_GE(finish - start, 3);
}

TEST(ConditionVariable, WaitForMilliseconds_Signalled) {
  thread::ThreadPool pool(Env::Default(), "test", 1);
  absl::Mutex m;
  absl::MutexLock l(&m);
  absl::CondVar cv;
  time_t start = time(nullptr);
  // Sleep for just 1 second then notify.  We have a timeout of 3 secs,
  // so the condition variable will notice the cv signal before the timeout.
  pool.Schedule([&m, &cv]() {
    Env::Default()->SleepForMicroseconds(1 * 1000 * 1000);
    absl::MutexLock l(&m);
    cv.SignalAll();
  });
  EXPECT_FALSE(cv.WaitWithTimeout(&m, absl::Milliseconds(3000)));
  time_t finish = time(nullptr);
  EXPECT_LT(finish - start, 3);
}

TEST(ConditionalCriticalSections, AwaitWithDeadline_Timeout) {
  bool always_false = false;
  absl::Mutex m;
  m.Lock();
  time_t start = time(nullptr);
  bool result = m.AwaitWithDeadline(
      absl::Condition(&always_false),
      absl::FromUnixNanos(EnvTime::NowNanos() + 3 * EnvTime::kSecondsToNanos));
  time_t finish = time(nullptr);
  m.Unlock();
  EXPECT_FALSE(result);
  EXPECT_GE(finish - start, 3);
}

TEST(ConditionalCriticalSections, AwaitWithDeadline_Woken) {
  thread::ThreadPool pool(Env::Default(), "test", 1);
  bool woken = false;
  absl::Mutex m;
  m.Lock();
  time_t start = time(nullptr);
  // Sleep for just 1 second then set the boolean.  We have a timeout of 3
  // secs, so the absl::Mutex implementation will notice the boolean state
  // change before the timeout.
  pool.Schedule([&m, &woken]() {
    Env::Default()->SleepForMicroseconds(1 * 1000 * 1000);
    m.Lock();
    woken = true;
    m.Unlock();
  });
  bool result = m.AwaitWithDeadline(
      absl::Condition(&woken),
      absl::FromUnixNanos(EnvTime::NowNanos() + 3 * EnvTime::kSecondsToNanos));
  time_t finish = time(nullptr);
  m.Unlock();
  EXPECT_EQ(result, true);
  EXPECT_LT(finish - start, 3);
}

// Return the negation of *b.  Used as an Await() predicate.
static bool Invert(bool* b) { return !*b; }

// The Value() method inverts the value of the boolean specified in
// the constructor.
class InvertClass {
 public:
  explicit InvertClass(bool* value) : value_(value) {}
  bool Value() { return !*this->value_; }

 private:
  InvertClass();
  bool* value_;
};

TEST(ConditionalCriticalSections, Await_PingPong) {
  thread::ThreadPool pool(Env::Default(), "test", 1);
  bool ping_pong = false;
  bool done = false;
  absl::Mutex m;
  pool.Schedule([&m, &ping_pong, &done]() {
    m.Lock();
    for (int i = 0; i != 1000; i++) {
      m.Await(absl::Condition(&ping_pong));
      ping_pong = false;
    }
    done = true;
    m.Unlock();
  });
  m.Lock();
  InvertClass invert(&ping_pong);
  for (int i = 0; i != 1000; i++) {
    m.Await(absl::Condition(&Invert, &ping_pong));
    ping_pong = true;
  }
  m.Await(absl::Condition(&done));
  m.Unlock();
}

TEST(ConditionalCriticalSections, Await_PingPongMethod) {
  thread::ThreadPool pool(Env::Default(), "test", 1);
  bool ping_pong = false;
  bool done = false;
  absl::Mutex m;
  pool.Schedule([&m, &ping_pong, &done]() {
    m.Lock();
    for (int i = 0; i != 1000; i++) {
      m.Await(absl::Condition(&ping_pong));
      ping_pong = false;
    }
    done = true;
    m.Unlock();
  });
  m.Lock();
  InvertClass invert(&ping_pong);
  for (int i = 0; i != 1000; i++) {
    m.Await(absl::Condition(&invert, &InvertClass::Value));
    ping_pong = true;
  }
  m.Await(absl::Condition(&done));
  m.Unlock();
}

TEST(TestCPUFeature, TestFeature) {
  // We don't know what the result should be on this platform, so just make
  // sure it's callable.
  const bool has_avx = TestCPUFeature(CPUFeature::AVX);
  LOG(INFO) << "has_avx = " << has_avx;
  const bool has_avx2 = TestCPUFeature(CPUFeature::AVX2);
  LOG(INFO) << "has_avx2 = " << has_avx2;
}

}  // namespace port
}  // namespace tsl
