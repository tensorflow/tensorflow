/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/thread_safe_buffer.h"

#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::UnorderedElementsAre;

TEST(ThreadSafeBufferTest, OneReaderAndOneWriter) {
  ThreadSafeBuffer<Tensor> buffer(/*buffer_size=*/1);
  auto thread = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"writer_thread",
      [&buffer]() { TF_EXPECT_OK(buffer.Push(Tensor("Test tensor"))); }));

  TF_ASSERT_OK_AND_ASSIGN(Tensor tensor, buffer.Pop());
  test::ExpectEqual(tensor, Tensor("Test tensor"));
}

TEST(ThreadSafeBufferTest, OneReaderAndMultipleWriters) {
  constexpr size_t kNumOfElements = 10;
  ThreadSafeBuffer<int> buffer(/*buffer_size=*/1);
  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < kNumOfElements; ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("writer_thread_", i),
        [&buffer, i] { TF_EXPECT_OK(buffer.Push(i)); })));
  }

  std::vector<int> results;
  for (int i = 0; i < kNumOfElements; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
    results.push_back(next);
  }
  EXPECT_THAT(results, UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

TEST(ThreadSafeBufferTest, MultipleReadersAndOneWriter) {
  constexpr size_t kNumOfElements = 10;
  ThreadSafeBuffer<int> buffer(/*buffer_size=*/1);
  mutex mu;
  std::vector<int> results;

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < kNumOfElements; ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("reader_thread_", i),
        [&buffer, &mu, &results]() {
          TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
          mutex_lock l(mu);
          results.push_back(next);
        })));
  }

  for (int i = 0; i < kNumOfElements; ++i) {
    TF_EXPECT_OK(buffer.Push(i));
  }

  // Wait for all threads to complete.
  threads.clear();
  EXPECT_THAT(results, UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

TEST(ThreadSafeBufferTest, MultipleReadersAndWriters) {
  constexpr size_t kNumOfElements = 10;
  ThreadSafeBuffer<int> buffer(/*buffer_size=*/1);
  mutex mu;
  std::vector<int> results;

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < kNumOfElements; ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("reader_thread_", i),
        [&buffer, &mu, &results]() {
          TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
          mutex_lock l(mu);
          results.push_back(next);
        })));
  }

  for (int i = 0; i < kNumOfElements; ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("writer_thread_", i),
        [&buffer, i]() { TF_EXPECT_OK(buffer.Push(i)); })));
  }

  // Wait for all threads to complete.
  threads.clear();
  EXPECT_THAT(results, UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

TEST(ThreadSafeBufferTest, CancelReaders) {
  ThreadSafeBuffer<int> buffer(/*buffer_size=*/1);
  std::vector<std::unique_ptr<Thread>> threads;

  for (int i = 0; i < 10; ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("reader_thread_", i),
        [&buffer]() {
          EXPECT_TRUE(errors::IsAborted(buffer.Pop().status()));
        })));
  }
  buffer.Cancel(errors::Aborted("Aborted"));
}

TEST(ThreadSafeBufferTest, CancelWriters) {
  constexpr size_t kNumOfElements = 10;
  ThreadSafeBuffer<Tensor> buffer(/*buffer_size=*/1);
  TF_EXPECT_OK(buffer.Push(Tensor("Test tensor")));

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < kNumOfElements; ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("writer_thread_", i),
        [&buffer]() {
          for (int i = 0; i < 100; ++i) {
            EXPECT_TRUE(
                errors::IsCancelled(buffer.Push(Tensor("Test tensor"))));
          }
        })));
  }
  buffer.Cancel(errors::Cancelled("Cancelled"));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
