/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/data/unbounded_thread_pool.h"

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

TEST(UnboundedThreadPool, ConcurrentThreadCreation) {
  UnboundedThreadPool pool(Env::Default(), "test");
  auto thread_factory = pool.get_thread_factory();

  // Create ten threads that each create ten threads that update a variable, and
  // ensure that they all run to completion.
  std::vector<std::unique_ptr<Thread>> threads;
  const int kNumThreadsToCreate = 10;
  std::atomic<int> i(0);
  for (int j = 0; j < kNumThreadsToCreate; ++j) {
    threads.push_back(thread_factory->StartThread("", [=, &i,
                                                       &thread_factory]() {
      std::vector<std::unique_ptr<Thread>> nested_threads;
      for (int k = 0; k < kNumThreadsToCreate; ++k) {
        nested_threads.push_back(
            thread_factory->StartThread("", [&i]() { ++i; }));
      }
      nested_threads.clear();
    }));
  }
  threads.clear();

  EXPECT_EQ(i, kNumThreadsToCreate * kNumThreadsToCreate);
}

TEST(UnboundedThreadPool, MultipleBlockingThreads) {
  UnboundedThreadPool pool(Env::Default(), "test");
  auto thread_factory = pool.get_thread_factory();

  std::vector<std::unique_ptr<Thread>> threads;

  // Create multiple waves (with increasing sizes) of threads that all block
  // before returning, and ensure that we terminate correctly.
  std::vector<int> round_sizes = {5, 10, 15, 20};

  for (const int round_size : round_sizes) {
    Notification n;
    BlockingCounter bc(round_size);
    for (int j = 0; j < round_size; ++j) {
      threads.push_back(thread_factory->StartThread("", [&bc, &n]() {
        bc.DecrementCount();
        // Block until `n` is notified, so that all ten threads must been
        // created before the first one completes.
        n.WaitForNotification();
      }));
    }

    // Wait until all threads have started. Since the number of threads in each
    // wave is increasing, we should have at least that number of threads in the
    // pool.
    bc.Wait();
    n.Notify();
    threads.clear();
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
