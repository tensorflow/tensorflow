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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/run_handler.h"

#include <memory>
#include <vector>

#define EIGEN_USE_THREADS
#include "absl/memory/memory.h"
#include "absl/synchronization/barrier.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(RunHandlerUtilTest, TestBasicScheduling) {
  int num_threads = 2;
  int num_handlers = 10;

  std::unique_ptr<RunHandlerPool> pool(new RunHandlerPool(num_threads));

  // RunHandler has 2 * num_threads (inter + intra) -
  // all should be able to run concurrently.
  absl::Barrier barrier1(num_threads);
  absl::Barrier barrier2(num_threads);

  BlockingCounter counter(2 * num_handlers * num_threads);

  int num_test_threads = 10;
  thread::ThreadPool test_pool(Env::Default(), "test", num_test_threads);
  for (int i = 0; i < 10; ++i) {
    test_pool.Schedule([&counter, &barrier1, &barrier2, &pool, i,
                        num_threads]() {
      auto handler = pool->Get();
      BlockingCounter local_counter(2 * num_threads);
      auto intra_thread_pool = handler->AsIntraThreadPoolInterface();

      for (int j = 0; j < num_threads; ++j) {
        handler->ScheduleInterOpClosure(
            [&local_counter, &counter, &barrier1, i]() {
              if (i == 2) {
                barrier1.Block();
              }
              counter.DecrementCount();
              local_counter.DecrementCount();
            });
        intra_thread_pool->Schedule([&local_counter, &counter, &barrier2, i]() {
          if (i == 9) {
            barrier2.Block();
          }
          counter.DecrementCount();
          local_counter.DecrementCount();
        });
      }
      local_counter.Wait();
    });
  }
  counter.Wait();
}

}  // namespace
}  // namespace tensorflow
