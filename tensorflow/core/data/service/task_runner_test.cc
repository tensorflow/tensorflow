/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/core/data/service/task_runner.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::testing::IsOkAndHolds;
using ::tensorflow::testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAreArray;

constexpr size_t kSmallCache = 100;                     // 100 bytes
constexpr size_t kLargeCache = 10 * (size_t{1} << 30);  // 10GB

class RangeIterator : public TaskIterator {
 public:
  explicit RangeIterator(const int64_t range, const bool repeat)
      : range_(range), repeat_(repeat) {}

  absl::Status GetNext(std::vector<Tensor>& element,
                       bool& end_of_sequence) override {
    end_of_sequence = (next_ >= range_);
    if (end_of_sequence) {
      return absl::OkStatus();
    }
    element = {Tensor{next_++}};
    if (repeat_) {
      next_ = next_ % range_;
    }
    return absl::OkStatus();
  }

  int64_t Cardinality() const override {
    return repeat_ ? kInfiniteCardinality : range_;
  }

 private:
  const int64_t range_;
  const bool repeat_;
  int64_t next_ = 0;
};

class InfiniteRangeIterator : public TaskIterator {
 public:
  InfiniteRangeIterator() = default;

  absl::Status GetNext(std::vector<Tensor>& element,
                       bool& end_of_sequence) override {
    element = {Tensor{next_++}};
    return absl::OkStatus();
  }

  int64_t Cardinality() const override { return kInfiniteCardinality; }

 private:
  int64_t next_ = 0;
};

template <class T>
class ElementOrErrorIterator : public TaskIterator {
 public:
  explicit ElementOrErrorIterator(const std::vector<StatusOr<T>>& elements)
      : elements_(elements) {}

  absl::Status GetNext(std::vector<Tensor>& element,
                       bool& end_of_sequence) override {
    end_of_sequence = (next_ >= elements_.size());
    if (end_of_sequence) {
      return absl::OkStatus();
    }
    const StatusOr<T>& next_element = elements_[next_++];
    TF_RETURN_IF_ERROR(next_element.status());
    element = {Tensor{*next_element}};
    return absl::OkStatus();
  }

  int64_t Cardinality() const override { return elements_.size(); }

 private:
  const std::vector<StatusOr<T>> elements_;
  int64_t next_ = 0;
};

template <class T>
StatusOr<std::vector<T>> GetTaskRunnerOutput(TaskRunner& runner,
                                             const GetElementRequest& request) {
  std::vector<T> output;
  for (bool end_of_sequence = false; !end_of_sequence;) {
    GetElementResult result;
    TF_RETURN_IF_ERROR(runner.GetNext(request, result));
    end_of_sequence = result.end_of_sequence;
    if (end_of_sequence) {
      break;
    }
    if (result.components.size() != 1) {
      return errors::Internal("GetElementResult Tensor size should be 1.");
    }
    output.push_back(result.components[0].unaligned_flat<T>().data()[0]);
  }
  return output;
}

template <class T>
StatusOr<T> GetNextFromTaskRunner(TaskRunner& runner,
                                  const GetElementRequest& request) {
  GetElementResult result;
  TF_RETURN_IF_ERROR(runner.GetNext(request, result));
  if (result.end_of_sequence) {
    return errors::OutOfRange("TaskRunner has reached the end of sequence.");
  }
  if (result.components.size() != 1) {
    return errors::Internal("GetElementResult Tensor size should be 1.");
  }
  return result.components[0].unaligned_flat<T>().data()[0];
}

template <class T>
StatusOr<std::vector<T>> GetElementsFromTaskRunner(
    TaskRunner& runner, const GetElementRequest& request,
    const size_t num_elements) {
  std::vector<T> output;
  for (size_t i = 0; i < num_elements; ++i) {
    TF_ASSIGN_OR_RETURN(T next, GetNextFromTaskRunner<T>(runner, request));
    output.push_back(next);
  }
  return output;
}

std::vector<int64_t> GetRange(const size_t range) {
  std::vector<int64_t> result;
  for (int64_t i = 0; i < range; ++i) {
    result.push_back(i);
  }
  return result;
}

// Reads from the task runner, storing results in `*output`.
absl::Status RunConsumer(int64_t consumer_index, int64_t start_index,
                         int64_t end_index, TaskRunner& task_runner,
                         std::vector<int64_t>& output) {
  for (int64_t next_index = start_index; next_index < end_index; ++next_index) {
    GetElementRequest request;
    request.set_round_index(next_index);
    request.set_consumer_index(consumer_index);
    request.set_skipped_previous_round(false);
    request.set_allow_skip(false);
    GetElementResult result;
    do {
      TF_RETURN_IF_ERROR(task_runner.GetNext(request, result));
      if (!result.end_of_sequence) {
        output.push_back(result.components[0].flat<int64_t>()(0));
      }
    } while (result.skip);
  }
  return absl::OkStatus();
}
}  // namespace

TEST(FirstComeFirstServedTaskRunnerTest, GetNext) {
  size_t range = 10;
  FirstComeFirstServedTaskRunner runner(
      std::make_unique<RangeIterator>(range, /*repeat=*/false));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<int64_t> output,
      GetTaskRunnerOutput<int64_t>(runner, GetElementRequest()));
  EXPECT_THAT(output, ElementsAreArray(GetRange(range)));

  GetElementResult result;
  TF_ASSERT_OK(runner.GetNext(GetElementRequest(), result));
  EXPECT_TRUE(result.end_of_sequence);
}

TEST(FirstComeFirstServedTaskRunnerTest, EmptyDataset) {
  FirstComeFirstServedTaskRunner runner(
      std::make_unique<RangeIterator>(/*range=*/0, /*repeat=*/false));

  for (int i = 0; i < 5; ++i) {
    GetElementResult result;
    TF_ASSERT_OK(runner.GetNext(GetElementRequest(), result));
    EXPECT_TRUE(result.end_of_sequence);
  }
}

TEST(FirstComeFirstServedTaskRunnerTest, Cancel) {
  size_t range = 10;
  FirstComeFirstServedTaskRunner runner(
      std::make_unique<RangeIterator>(range, /*repeat=*/false));
  runner.Cancel();

  for (int i = 0; i < range; ++i) {
    GetElementResult result;
    EXPECT_THAT(runner.GetNext(GetElementRequest(), result),
                testing::StatusIs(error::CANCELLED));
  }
}

TEST(FirstComeFirstServedTaskRunnerTest, ConcurrentReaders) {
  size_t range = 1000;
  size_t num_readers = 10;
  FirstComeFirstServedTaskRunner runner(
      std::make_unique<RangeIterator>(range, /*repeat=*/false));

  mutex mu;
  std::vector<int64_t> results;  // Guarded by `mu`.
  std::vector<std::unique_ptr<Thread>> reader_threads;
  for (int i = 0; i < num_readers; ++i) {
    reader_threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Trainer_", i),
        [&runner, &results, &mu]() {
          TF_ASSERT_OK_AND_ASSIGN(
              std::vector<int64_t> output,
              GetTaskRunnerOutput<int64_t>(runner, GetElementRequest()));

          GetElementResult result;
          TF_ASSERT_OK(runner.GetNext(GetElementRequest(), result));
          EXPECT_TRUE(result.end_of_sequence);

          mutex_lock l(mu);
          std::move(output.begin(), output.end(), std::back_inserter(results));
        })));
  }
  for (auto& thread : reader_threads) {
    thread.reset();
  }

  EXPECT_THAT(results, UnorderedElementsAreArray(GetRange(range)));
}

TEST(FirstComeFirstServedTaskRunnerTest, GetNextAndCancel) {
  size_t range = 10;
  FirstComeFirstServedTaskRunner runner(
      std::make_unique<RangeIterator>(range, /*repeat=*/false));

  int64_t i;
  for (i = 0; i < range / 2; ++i) {
    EXPECT_THAT(GetNextFromTaskRunner<int64_t>(runner, GetElementRequest()),
                IsOkAndHolds(i));
  }
  runner.Cancel();

  for (; i < range; ++i) {
    GetElementResult result;
    EXPECT_THAT(runner.GetNext(GetElementRequest(), result),
                testing::StatusIs(error::CANCELLED));
  }
}

TEST(FirstComeFirstServedTaskRunnerTest, Error) {
  FirstComeFirstServedTaskRunner runner(
      std::make_unique<ElementOrErrorIterator<tstring>>(
          std::vector<absl::StatusOr<tstring>>{
              tstring("First element"),
              errors::InvalidArgument("Invalid argument"),
              tstring("Second element"), errors::Aborted("Aborted")}));
  EXPECT_THAT(GetNextFromTaskRunner<tstring>(runner, GetElementRequest()),
              IsOkAndHolds("First element"));
  EXPECT_THAT(GetNextFromTaskRunner<tstring>(runner, GetElementRequest()),
              testing::StatusIs(error::INVALID_ARGUMENT));
  EXPECT_THAT(GetNextFromTaskRunner<tstring>(runner, GetElementRequest()),
              IsOkAndHolds("Second element"));
  EXPECT_THAT(GetNextFromTaskRunner<tstring>(runner, GetElementRequest()),
              testing::StatusIs(error::ABORTED));
}

TEST(CachingTaskRunnerTest, GetNext) {
  size_t range = 10;
  CachingTaskRunner runner(std::make_unique<InfiniteRangeIterator>(),
                           /*max_cache_size_bytes=*/kLargeCache);

  size_t num_trainers = 10;
  for (size_t i = 0; i < num_trainers; ++i) {
    GetElementRequest request;
    request.set_trainer_id(absl::StrCat("Trainer ", i));
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<int64_t> output,
        GetElementsFromTaskRunner<int64_t>(runner, request, range));
    EXPECT_THAT(output, ElementsAreArray(GetRange(range)));

    GetElementResult result;
    TF_ASSERT_OK(runner.GetNext(request, result));
    EXPECT_FALSE(result.end_of_sequence);
  }
}

TEST(CachingTaskRunnerTest, EmptyDataset) {
  CachingTaskRunner runner(
      std::make_unique<RangeIterator>(/*range=*/0, /*repeat=*/false),
      /*max_cache_size_bytes=*/kLargeCache);
  GetElementRequest request;
  request.set_trainer_id("Trainer ID");

  GetElementResult result;
  EXPECT_THAT(runner.GetNext(request, result),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Cross-trainer caching requires the input "
                                 "dataset to be infinite.")));
}

TEST(CachingTaskRunnerTest, SlowClientSkipsData) {
  size_t range = 1000;
  CachingTaskRunner runner(std::make_unique<InfiniteRangeIterator>(),
                           /*max_cache_size_bytes=*/kSmallCache);

  GetElementRequest request;
  request.set_trainer_id("Fast trainer");
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<int64_t> fast_trainer_output,
      GetElementsFromTaskRunner<int64_t>(runner, request, range));
  EXPECT_THAT(fast_trainer_output, ElementsAreArray(GetRange(range)));

  request.set_trainer_id("Slow trainer");
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<int64_t> slow_trainer_output,
      GetElementsFromTaskRunner<int64_t>(runner, request, range));
  EXPECT_THAT(slow_trainer_output, SizeIs(range));
  EXPECT_THAT(slow_trainer_output[0], Gt(0));
}

TEST(CachingTaskRunnerTest, ConcurrentTrainers) {
  size_t range = 100;
  size_t num_readers = 10;
  CachingTaskRunner runner(std::make_unique<InfiniteRangeIterator>(),
                           /*max_cache_size_bytes=*/kLargeCache);

  // When the cache is large enough, every trainer can read all the elements.
  std::vector<std::unique_ptr<Thread>> reader_threads;
  for (int i = 0; i < num_readers; ++i) {
    reader_threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Trainer_", i),
        [&runner, range, i]() {
          GetElementRequest request;
          request.set_trainer_id(absl::StrCat("Trainer_", i));
          TF_ASSERT_OK_AND_ASSIGN(
              std::vector<int64_t> output,
              GetElementsFromTaskRunner<int64_t>(runner, request, range));
          EXPECT_THAT(output, ElementsAreArray(GetRange(range)));

          GetElementResult result;
          TF_ASSERT_OK(runner.GetNext(request, result));
          EXPECT_FALSE(result.end_of_sequence);
        })));
  }
}

TEST(CachingTaskRunnerTest, Cancel) {
  CachingTaskRunner runner(std::make_unique<InfiniteRangeIterator>(),
                           /*max_cache_size_bytes=*/kLargeCache);

  GetElementRequest request;
  request.set_trainer_id("Trainer ID");
  int i;
  for (i = 0; i < 10; ++i) {
    EXPECT_THAT(GetNextFromTaskRunner<int64_t>(runner, request),
                IsOkAndHolds(i));
  }
  runner.Cancel();

  for (; i < 10; ++i) {
    GetElementResult result;
    EXPECT_THAT(runner.GetNext(request, result),
                testing::StatusIs(error::CANCELLED));
  }
}

TEST(CachingTaskRunnerTest, CancelConcurrentReaders) {
  size_t num_readers = 10;
  CachingTaskRunner runner(std::make_unique<InfiniteRangeIterator>(),
                           /*max_cache_size_bytes=*/kSmallCache);

  // The readers keep getting elements until cancelled.
  std::vector<std::unique_ptr<Thread>> reader_threads;
  for (size_t i = 0; i < num_readers; ++i) {
    reader_threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Trainer_", i),
        [&runner]() {
          for (size_t j = 0; true; ++j) {
            GetElementRequest request;
            request.set_trainer_id(absl::StrCat("Trainer_", (j % 100)));
            GetElementResult result;
            absl::Status status = runner.GetNext(request, result);
            if (!status.ok()) {
              return;
            }
            ASSERT_FALSE(result.end_of_sequence);
            ASSERT_EQ(result.components.size(), 1);
          }
        })));
  }

  Env::Default()->SleepForMicroseconds(1000000);
  runner.Cancel();
  for (auto& thread : reader_threads) {
    thread.reset();
  }

  GetElementRequest request;
  GetElementResult result;
  request.set_trainer_id(absl::StrCat("Trainer_", 0));
  EXPECT_THAT(runner.GetNext(request, result),
              testing::StatusIs(error::CANCELLED));
}

TEST(CachingTaskRunnerTest, Errors) {
  size_t num_readers = 10;
  CachingTaskRunner runner(
      std::make_unique<ElementOrErrorIterator<tstring>>(
          std::vector<absl::StatusOr<tstring>>{
              tstring("First element"),
              errors::Cancelled("Cancelled"),
              tstring("Second element"),
              errors::FailedPrecondition("FailedPrecondition"),
              tstring("Third element"),
              errors::Unavailable("Unavailable"),
          }),
      /*max_cache_size_bytes=*/kLargeCache);

  std::vector<std::unique_ptr<Thread>> reader_threads;
  std::vector<std::vector<tstring>> results;
  results.reserve(num_readers);
  for (size_t i = 0; i < num_readers; ++i) {
    results.emplace_back();
    std::vector<tstring>& result = results.back();
    reader_threads.push_back(absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/absl::StrCat("Trainer_", i),
        [&runner, &result, i]() {
          GetElementRequest request;
          request.set_trainer_id(absl::StrCat("Trainer_", i));
          while (true) {
            absl::StatusOr<tstring> element =
                GetNextFromTaskRunner<tstring>(runner, request);
            if (element.ok()) {
              result.push_back(*element);
            }
            if (absl::IsInvalidArgument(element.status())) {
              EXPECT_THAT(
                  element.status(),
                  StatusIs(error::INVALID_ARGUMENT,
                           HasSubstr("Cross-trainer caching requires the input "
                                     "dataset to be infinite.")));
              return;
            }
          }
        })));
  }
  for (auto& thread : reader_threads) {
    thread.reset();
  }

  // The readers can read the non-error elements.
  EXPECT_EQ(results.size(), num_readers);
  for (const std::vector<tstring>& result : results) {
    EXPECT_THAT(result,
                ElementsAre(tstring("First element"), tstring("Second element"),
                            tstring("Third element")));
  }
}

class ConsumeParallelTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<int64_t, int64_t>> {};

TEST_P(ConsumeParallelTest, ConsumeParallel) {
  int64_t num_elements = std::get<0>(GetParam());
  int64_t num_consumers = std::get<1>(GetParam());
  RoundRobinTaskRunner runner(
      std::make_unique<RangeIterator>(num_elements, /*repeat=*/true),
      num_consumers,
      /*worker_address=*/"test_worker_address");
  std::vector<std::vector<int64_t>> per_consumer_results;
  std::vector<std::unique_ptr<Thread>> consumers;
  mutex mu;
  absl::Status error;
  for (int consumer = 0; consumer < num_consumers; ++consumer) {
    mutex_lock l(mu);
    per_consumer_results.emplace_back();
    consumers.push_back(absl::WrapUnique(Env::Default()->StartThread(
        {}, absl::StrCat("consumer_", consumer), [&, consumer] {
          std::vector<int64_t> results;
          absl::Status s =
              RunConsumer(consumer, /*start_index=*/0,
                          /*end_index=*/num_elements, runner, results);
          mutex_lock l(mu);
          if (!s.ok()) {
            error = s;
            return;
          }
          per_consumer_results[consumer] = std::move(results);
        })));
  }
  // Wait for all consumers to finish;
  consumers.clear();
  mutex_lock l(mu);
  TF_ASSERT_OK(error);
  for (int i = 0; i < num_elements; ++i) {
    int consumer = i % num_consumers;
    int round = i / num_consumers;
    EXPECT_EQ(per_consumer_results[consumer][round], i);
  }
}

INSTANTIATE_TEST_SUITE_P(ConsumeParallelTests, ConsumeParallelTest,
                         // tuples represent <num_elements, num_consumers>
                         ::testing::Values(std::make_tuple(1000, 5),
                                           std::make_tuple(1003, 5),
                                           std::make_tuple(1000, 20),
                                           std::make_tuple(4, 20),
                                           std::make_tuple(0, 20)));

TEST(RoundRobinTaskRunner, ConsumeParallelPartialRound) {
  int64_t num_consumers = 5;
  std::vector<int64_t> starting_rounds = {12, 11, 11, 12, 12};
  int64_t end_index = 15;
  std::vector<std::vector<int64_t>> expected_consumer_results = {
      {5, 10, 15}, {1, 6, 11, 16}, {2, 7, 12, 17}, {8, 13, 18}, {9, 14, 19}};
  RoundRobinTaskRunner runner(
      std::make_unique<RangeIterator>(30, /*repeat=*/true), num_consumers,
      /*worker_address=*/"test_worker_address");
  std::vector<std::vector<int64_t>> per_consumer_results;
  std::vector<std::unique_ptr<Thread>> consumers;
  mutex mu;
  absl::Status error;
  for (int consumer = 0; consumer < num_consumers; ++consumer) {
    mutex_lock l(mu);
    per_consumer_results.emplace_back();
    consumers.push_back(absl::WrapUnique(Env::Default()->StartThread(
        {}, absl::StrCat("consumer_", consumer), [&, consumer] {
          std::vector<int64_t> results;
          absl::Status s = RunConsumer(consumer, starting_rounds[consumer],
                                       end_index, runner, results);
          mutex_lock l(mu);
          if (!s.ok()) {
            error = s;
            return;
          }
          per_consumer_results[consumer] = std::move(results);
        })));
  }
  // Wait for all consumers to finish;
  consumers.clear();
  mutex_lock l(mu);
  TF_ASSERT_OK(error);
  for (int consumer = 0; consumer < num_consumers; ++consumer) {
    EXPECT_EQ(per_consumer_results[consumer],
              expected_consumer_results[consumer]);
  }
}
}  // namespace data
}  // namespace tensorflow
