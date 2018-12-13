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

#include <condition_variable>
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
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
  EXPECT_GE(cpu, 0);
  EXPECT_LT(cpu, NumTotalCPUs());
}

TEST(ConditionVariable, WaitForMilliseconds_Timeout) {
  mutex m;
  mutex_lock l(m);
  condition_variable cv;
  ConditionResult result = kCond_MaybeNotified;
  time_t start = time(nullptr);
  // Condition variables are subject to spurious wakeups on some platforms,
  // so need to check for a timeout within a loop.
  while (result == kCond_MaybeNotified) {
    result = WaitForMilliseconds(&l, &cv, 3000);
  }
  EXPECT_EQ(result, kCond_Timeout);
  time_t finish = time(nullptr);
  EXPECT_GE(finish - start, 3);
}

TEST(ConditionVariable, WaitForMilliseconds_Signalled) {
  thread::ThreadPool pool(Env::Default(), "test", 1);
  mutex m;
  mutex_lock l(m);
  condition_variable cv;
  time_t start = time(nullptr);
  // Sleep for just 1 second then notify.  We have a timeout of 3 secs,
  // so the condition variable will notice the cv signal before the timeout.
  pool.Schedule([&m, &cv]() {
    Env::Default()->SleepForMicroseconds(1 * 1000 * 1000);
    mutex_lock l(m);
    cv.notify_all();
  });
  EXPECT_EQ(WaitForMilliseconds(&l, &cv, 3000), kCond_MaybeNotified);
  time_t finish = time(nullptr);
  EXPECT_LT(finish - start, 3);
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
}  // namespace tensorflow

int main(int argc, char** argv) {
  // On Linux, add: FLAGS_logtostderr = true;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
