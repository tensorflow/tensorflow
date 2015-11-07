#include "tensorflow/core/lib/core/threadpool.h"

#include <atomic>

#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/env.h"
#include <gtest/gtest.h>

namespace tensorflow {
namespace thread {

static const int kNumThreads = 30;

TEST(ThreadPool, Empty) {
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    ThreadPool pool(Env::Default(), "test", num_threads);
  }
}

TEST(ThreadPool, DoWork) {
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    const int kWorkItems = 15;
    bool work[kWorkItems];
    for (int i = 0; i < kWorkItems; i++) {
      work[i] = false;
    }
    {
      ThreadPool pool(Env::Default(), "test", num_threads);
      for (int i = 0; i < kWorkItems; i++) {
        pool.Schedule([&work, i]() {
          ASSERT_FALSE(work[i]);
          work[i] = true;
        });
      }
    }
    for (int i = 0; i < kWorkItems; i++) {
      ASSERT_TRUE(work[i]);
    }
  }
}

static void BM_Sequential(int iters) {
  ThreadPool pool(Env::Default(), "test", kNumThreads);
  // Decrement count sequentially until 0.
  int count = iters;
  mutex done_lock;
  condition_variable done;
  bool done_flag = false;
  std::function<void()> work = [&pool, &count, &done_lock, &done, &done_flag,
                                &work]() {
    if (count--) {
      pool.Schedule(work);
    } else {
      mutex_lock l(done_lock);
      done_flag = true;
      done.notify_all();
    }
  };
  work();
  mutex_lock l(done_lock);
  if (!done_flag) {
    done.wait(l);
  }
}
BENCHMARK(BM_Sequential);

static void BM_Parallel(int iters) {
  ThreadPool pool(Env::Default(), "test", kNumThreads);
  // Decrement count concurrently until 0.
  std::atomic_int_fast32_t count(iters);
  mutex done_lock;
  condition_variable done;
  bool done_flag = false;
  for (int i = 0; i < iters; ++i) {
    pool.Schedule([&count, &done_lock, &done, &done_flag]() {
      if (count.fetch_sub(1) == 1) {
        mutex_lock l(done_lock);
        done_flag = true;
        done.notify_all();
      }
    });
  }
  mutex_lock l(done_lock);
  if (!done_flag) {
    done.wait(l);
  }
}
BENCHMARK(BM_Parallel);

}  // namespace thread
}  // namespace tensorflow
