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
#include <tuple>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::testing::IsOk;
using ::tensorflow::testing::StatusIs;
using ::testing::UnorderedElementsAreArray;

class ThreadSafeBufferTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<size_t, size_t>> {
 protected:
  size_t GetBufferSize() const { return std::get<0>(GetParam()); }
  size_t GetNumOfElements() const { return std::get<1>(GetParam()); }
};

std::vector<int> GetRange(const size_t range) {
  std::vector<int> result;
  for (int i = 0; i < range; ++i) {
    result.push_back(i);
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(VaryingBufferAndInputSizes, ThreadSafeBufferTest,
                         ::testing::Values(std::make_tuple(1, 2),
                                           std::make_tuple(2, 10),
                                           std::make_tuple(10, 2)));

TEST_P(ThreadSafeBufferTest, OneReaderAndOneWriter) {
  ThreadSafeBuffer<int> buffer(GetBufferSize());
  auto thread = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"writer_thread", [this, &buffer]() {
        for (int i = 0; i < GetNumOfElements(); ++i) {
          ASSERT_THAT(buffer.Push(i), IsOk());
        }
      }));

  for (size_t i = 0; i < GetNumOfElements(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
    EXPECT_EQ(next, i);
  }
}

TEST_P(ThreadSafeBufferTest, OneReaderAndMultipleWriters) {
  ThreadSafeBuffer<int> buffer(GetBufferSize());
  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("writer_thread_", i),
        [&buffer, i] { ASSERT_THAT(buffer.Push(i), IsOk()); })));
  }

  std::vector<int> results;
  for (int i = 0; i < GetNumOfElements(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
    results.push_back(next);
  }
  EXPECT_THAT(results, UnorderedElementsAreArray(GetRange(GetNumOfElements())));
}

TEST_P(ThreadSafeBufferTest, MultipleReadersAndOneWriter) {
  ThreadSafeBuffer<int> buffer(GetBufferSize());
  mutex mu;
  std::vector<int> results;

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("reader_thread_", i),
        [&buffer, &mu, &results]() {
          TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
          mutex_lock l(mu);
          results.push_back(next);
        })));
  }

  for (int i = 0; i < GetNumOfElements(); ++i) {
    ASSERT_THAT(buffer.Push(i), IsOk());
  }

  // Wait for all threads to complete.
  threads.clear();
  EXPECT_THAT(results, UnorderedElementsAreArray(GetRange(GetNumOfElements())));
}

TEST_P(ThreadSafeBufferTest, MultipleReadersAndWriters) {
  ThreadSafeBuffer<int> buffer(GetBufferSize());
  mutex mu;
  std::vector<int> results;

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("reader_thread_", i),
        [&buffer, &mu, &results]() {
          TF_ASSERT_OK_AND_ASSIGN(int next, buffer.Pop());
          mutex_lock l(mu);
          results.push_back(next);
        })));
  }

  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("writer_thread_", i),
        [&buffer, i]() { ASSERT_THAT(buffer.Push(i), IsOk()); })));
  }

  // Wait for all threads to complete.
  threads.clear();
  EXPECT_THAT(results, UnorderedElementsAreArray(GetRange(GetNumOfElements())));
}

TEST_P(ThreadSafeBufferTest, BlockReaderWhenBufferIsEmpty) {
  ThreadSafeBuffer<Tensor> buffer(GetBufferSize());

  // The buffer is empty, blocking the next `Pop` call.
  auto thread = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"reader_thread", [&buffer]() {
        TF_ASSERT_OK_AND_ASSIGN(Tensor tensor, buffer.Pop());
        test::ExpectEqual(tensor, Tensor("Test tensor"));
      }));

  // Pushing an element unblocks the `Pop` call.
  Env::Default()->SleepForMicroseconds(10000);
  ASSERT_THAT(buffer.Push(Tensor("Test tensor")), IsOk());
}

TEST_P(ThreadSafeBufferTest, BlockWriterWhenBufferIsFull) {
  ThreadSafeBuffer<Tensor> buffer(GetBufferSize());
  // Fills the buffer to block the next `Push` call.
  for (int i = 0; i < GetBufferSize(); ++i) {
    ASSERT_THAT(buffer.Push(Tensor("Test tensor")), IsOk());
  }

  uint64 push_time = 0;
  auto thread = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"writer_thread", [&buffer, &push_time]() {
        ASSERT_THAT(buffer.Push(Tensor("Test tensor")), IsOk());
        push_time = Env::Default()->NowMicros();
      }));

  // Popping an element unblocks the `Push` call.
  Env::Default()->SleepForMicroseconds(10000);
  uint64 pop_time = Env::Default()->NowMicros();
  ASSERT_THAT(buffer.Pop(), IsOk());
  thread.reset();
  EXPECT_LE(pop_time, push_time);
}

TEST_P(ThreadSafeBufferTest, CancelReaders) {
  ThreadSafeBuffer<int> buffer(GetBufferSize());
  std::vector<std::unique_ptr<Thread>> threads;

  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("reader_thread_", i),
        [&buffer]() { EXPECT_THAT(buffer.Pop(), StatusIs(error::ABORTED)); })));
  }
  buffer.Cancel(errors::Aborted("Aborted"));
}

TEST_P(ThreadSafeBufferTest, CancelWriters) {
  ThreadSafeBuffer<Tensor> buffer(GetBufferSize());
  // Fills the buffer so subsequent pushes are all cancelled.
  for (int i = 0; i < GetBufferSize(); ++i) {
    ASSERT_THAT(buffer.Push(Tensor("Test tensor")), IsOk());
  }

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < GetNumOfElements(); ++i) {
    threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("writer_thread_", i),
        [&buffer]() {
          for (int i = 0; i < 100; ++i) {
            EXPECT_THAT(buffer.Push(Tensor("Test tensor")),
                        StatusIs(error::CANCELLED));
          }
        })));
  }
  buffer.Cancel(errors::Cancelled("Cancelled"));
}

TEST_P(ThreadSafeBufferTest, CancelMultipleTimes) {
  ThreadSafeBuffer<Tensor> buffer(GetBufferSize());
  buffer.Cancel(errors::Unknown("Unknown"));
  EXPECT_THAT(buffer.Push(Tensor("Test tensor")), StatusIs(error::UNKNOWN));
  buffer.Cancel(errors::DeadlineExceeded("Deadline exceeded"));
  EXPECT_THAT(buffer.Pop(), StatusIs(error::DEADLINE_EXCEEDED));
  buffer.Cancel(errors::ResourceExhausted("Resource exhausted"));
  EXPECT_THAT(buffer.Push(Tensor("Test tensor")),
              StatusIs(error::RESOURCE_EXHAUSTED));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
