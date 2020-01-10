#include "tensorflow/core/platform/port.h"
#include <condition_variable>
#include "tensorflow/core/lib/core/threadpool.h"
#include <gtest/gtest.h>

namespace tensorflow {
namespace port {

TEST(Port, AlignedMalloc) {
  for (size_t alignment = 1; alignment <= 1 << 20; alignment <<= 1) {
    void* p = aligned_malloc(1, alignment);
    ASSERT_TRUE(p != NULL) << "aligned_malloc(1, " << alignment << ")";
    uintptr_t pval = reinterpret_cast<uintptr_t>(p);
    EXPECT_EQ(pval % alignment, 0);
    aligned_free(p);
  }
}

TEST(ConditionVariable, WaitForMilliseconds_Timeout) {
  mutex m;
  mutex_lock l(m);
  condition_variable cv;
  time_t start = time(NULL);
  EXPECT_EQ(WaitForMilliseconds(&l, &cv, 3000), kCond_Timeout);
  time_t finish = time(NULL);
  EXPECT_GE(finish - start, 3);
}

TEST(ConditionVariable, WaitForMilliseconds_Signalled) {
  thread::ThreadPool pool(Env::Default(), "test", 1);
  mutex m;
  mutex_lock l(m);
  condition_variable cv;
  time_t start = time(NULL);
  // Sleep for just 1 second then notify.  We have a timeout of 3 secs,
  // so the condition variable will notice the cv signal before the timeout.
  pool.Schedule([&m, &cv]() {
    sleep(1);
    mutex_lock l(m);
    cv.notify_all();
  });
  EXPECT_EQ(WaitForMilliseconds(&l, &cv, 3000), kCond_MaybeNotified);
  time_t finish = time(NULL);
  EXPECT_LT(finish - start, 3);
}

}  // namespace port
}  // namespace tensorflow
