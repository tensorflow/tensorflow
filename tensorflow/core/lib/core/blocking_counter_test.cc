#include <gtest/gtest.h>

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace {

TEST(BlockingCounterTest, TestZero) {
  BlockingCounter bc(0);
  bc.Wait();
}

TEST(BlockingCounterTest, TestSingleThread) {
  BlockingCounter bc(2);
  bc.DecrementCount();
  bc.DecrementCount();
  bc.Wait();
}

TEST(BlockingCounterTest, TestMultipleThread) {
  int N = 3;
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", N);

  BlockingCounter bc(N);
  for (int i = 0; i < N; ++i) {
    thread_pool->Schedule([&bc] { bc.DecrementCount(); });
  }

  bc.Wait();
  delete thread_pool;
}

}  // namespace
}  // namespace tensorflow
