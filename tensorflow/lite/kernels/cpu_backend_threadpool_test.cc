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

#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/cpu_backend_context.h"

namespace tflite {

namespace {

class TestGenerateArrayOfIncrementingIntsTask
    : public cpu_backend_threadpool::Task {
 public:
  TestGenerateArrayOfIncrementingIntsTask(int* buffer, int start, int end)
      : buffer_(buffer), start_(start), end_(end) {}

  void Run() override {
    for (int i = start_; i < end_; i++) {
      buffer_[i] = i;
    }
  }

 private:
  int* buffer_;
  int start_;
  int end_;
};

void TestGenerateArrayOfIncrementingInts(int num_threads, int size) {
  // The buffer that our threads will write to.
  std::vector<int> buffer(size);

  // The tasks that our threads will run.
  std::vector<TestGenerateArrayOfIncrementingIntsTask> tasks;

  // Create task objects.
  int rough_size_per_thread = size / num_threads;
  int start = 0;
  for (int thread = 0; thread < num_threads; thread++) {
    int end = start + rough_size_per_thread;
    if (thread == num_threads - 1) {
      end = size;
    }
    tasks.emplace_back(buffer.data(), start, end);
    start = end;
  }
  ASSERT_EQ(num_threads, tasks.size());

  CpuBackendContext context;
  // This SetMaxNumThreads is only to satisfy an assertion in Execute.
  // What actually determines the number of threads used is the parameter
  // passed to Execute, since Execute does 1:1 mapping of tasks to threads.
  context.SetMaxNumThreads(num_threads);

  // Execute tasks on the threadpool.
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(), &context);

  // Check contents of the generated buffer.
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(buffer[i], i);
  }
}

TEST(CpuBackendThreadpoolTest, OneThreadSize100) {
  TestGenerateArrayOfIncrementingInts(1, 100);
}

TEST(CpuBackendThreadpoolTest, ThreeThreadsSize1000000) {
  TestGenerateArrayOfIncrementingInts(3, 1000000);
}

TEST(CpuBackendThreadpoolTest, TenThreadsSize1234567) {
  TestGenerateArrayOfIncrementingInts(10, 1234567);
}

}  // namespace

}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
