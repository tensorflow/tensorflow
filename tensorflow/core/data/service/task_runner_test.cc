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

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/data/dataset.pb.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

class TestTaskIterator : public TaskIterator {
 public:
  explicit TestTaskIterator(const std::vector<std::vector<Tensor>>& elements,
                            const bool repeat)
      : elements_(elements), index_(0), repeat_(repeat) {}

  Status GetNext(std::vector<Tensor>& element, bool& end_of_sequence) override {
    end_of_sequence = index_ >= elements_.size();
    if (!end_of_sequence) {
      element = elements_[index_];
      ++index_;
      if (repeat_) {
        index_ = index_ % elements_.size();
      }
    }
    return Status::OK();
  }

  int64_t Cardinality() const override { return kInfiniteCardinality; }

 private:
  std::vector<std::vector<Tensor>> elements_;
  int64_t index_;
  const bool repeat_;
};

class TestErrorIterator : public TaskIterator {
 public:
  explicit TestErrorIterator(Status status) : status_(std::move(status)) {}

  Status GetNext(std::vector<Tensor>& element, bool& end_of_sequence) override {
    return status_;
  }

  int64_t Cardinality() const override { return kInfiniteCardinality; }

 private:
  const Status status_;
};

std::vector<std::vector<Tensor>> GetRangeDataset(const size_t range) {
  std::vector<std::vector<Tensor>> dataset;
  for (int64_t i = 0; i < range; ++i) {
    dataset.push_back({Tensor(i)});
  }
  return dataset;
}

// Reads from the task runner, storing results in `*output`.
Status RunConsumer(int64_t consumer_index, int64_t start_index,
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
  return Status::OK();
}
}  // namespace

TEST(FirstComeFirstServedTaskRunnerTest, GetNext) {
  std::vector<std::vector<Tensor>> elements = GetRangeDataset(10);
  FirstComeFirstServedTaskRunner runner(
      absl::make_unique<TestTaskIterator>(elements, /*repeat=*/false));
  for (auto& expected_element : elements) {
    GetElementResult result;
    TF_ASSERT_OK(runner.GetNext(GetElementRequest(), result));
    ASSERT_FALSE(result.end_of_sequence);
    ASSERT_EQ(result.components.size(), 1);
    test::ExpectEqual(result.components[0], expected_element[0]);
  }

  GetElementResult result;
  TF_ASSERT_OK(runner.GetNext(GetElementRequest(), result));
  EXPECT_TRUE(result.end_of_sequence);
}

TEST(FirstComeFirstServedTaskRunnerTest, EmptyDataset) {
  std::vector<std::vector<Tensor>> elements;
  FirstComeFirstServedTaskRunner runner(
      absl::make_unique<TestTaskIterator>(elements, /*repeat=*/false));
  for (int i = 0; i < 5; ++i) {
    GetElementResult result;
    TF_ASSERT_OK(runner.GetNext(GetElementRequest(), result));
    EXPECT_TRUE(result.end_of_sequence);
  }
}

TEST(FirstComeFirstServedTaskRunnerTest, Cancel) {
  std::vector<std::vector<Tensor>> elements = GetRangeDataset(10);
  FirstComeFirstServedTaskRunner runner(
      absl::make_unique<TestTaskIterator>(elements, /*repeat=*/false));
  runner.Cancel();

  for (int i = 0; i < elements.size(); ++i) {
    GetElementResult result;
    EXPECT_THAT(runner.GetNext(GetElementRequest(), result),
                testing::StatusIs(error::CANCELLED));
  }
}

TEST(FirstComeFirstServedTaskRunnerTest, GetNextAndCancel) {
  std::vector<std::vector<Tensor>> elements = GetRangeDataset(10);
  FirstComeFirstServedTaskRunner runner(
      absl::make_unique<TestTaskIterator>(elements, /*repeat=*/false));

  int i;
  for (i = 0; i < elements.size() / 2; ++i) {
    GetElementResult result;
    TF_ASSERT_OK(runner.GetNext(GetElementRequest(), result));
    EXPECT_FALSE(result.end_of_sequence);
    ASSERT_EQ(result.components.size(), 1);
    test::ExpectEqual(result.components[0], elements[i][0]);
  }
  runner.Cancel();

  for (; i < elements.size(); ++i) {
    GetElementResult result;
    EXPECT_THAT(runner.GetNext(GetElementRequest(), result),
                testing::StatusIs(error::CANCELLED));
  }
}

TEST(FirstComeFirstServedTaskRunnerTest, Error) {
  FirstComeFirstServedTaskRunner runner(
      absl::make_unique<TestErrorIterator>(errors::Aborted("Aborted")));
  GetElementResult result;
  EXPECT_THAT(runner.GetNext(GetElementRequest(), result),
              testing::StatusIs(error::ABORTED));
  EXPECT_THAT(runner.GetNext(GetElementRequest(), result),
              testing::StatusIs(error::ABORTED));
  EXPECT_THAT(runner.GetNext(GetElementRequest(), result),
              testing::StatusIs(error::ABORTED));
}

class ConsumeParallelTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<int64_t, int64_t>> {};

TEST_P(ConsumeParallelTest, ConsumeParallel) {
  int64_t num_elements = std::get<0>(GetParam());
  int64_t num_consumers = std::get<1>(GetParam());
  std::vector<std::vector<Tensor>> elements = GetRangeDataset(num_elements);
  RoundRobinTaskRunner runner(
      absl::make_unique<TestTaskIterator>(elements, /*repeat=*/true),
      num_consumers,
      /*worker_address=*/"test_worker_address");
  std::vector<std::vector<int64_t>> per_consumer_results;
  std::vector<std::unique_ptr<Thread>> consumers;
  mutex mu;
  Status error;
  for (int consumer = 0; consumer < num_consumers; ++consumer) {
    mutex_lock l(mu);
    per_consumer_results.emplace_back();
    consumers.push_back(absl::WrapUnique(Env::Default()->StartThread(
        {}, absl::StrCat("consumer_", consumer), [&, consumer] {
          std::vector<int64_t> results;
          Status s = RunConsumer(consumer, /*start_index=*/0,
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
  std::vector<std::vector<Tensor>> elements = GetRangeDataset(30);
  RoundRobinTaskRunner runner(
      absl::make_unique<TestTaskIterator>(elements, /*repeat=*/true),
      num_consumers,
      /*worker_address=*/"test_worker_address");
  std::vector<std::vector<int64_t>> per_consumer_results;
  std::vector<std::unique_ptr<Thread>> consumers;
  mutex mu;
  Status error;
  for (int consumer = 0; consumer < num_consumers; ++consumer) {
    mutex_lock l(mu);
    per_consumer_results.emplace_back();
    consumers.push_back(absl::WrapUnique(Env::Default()->StartThread(
        {}, absl::StrCat("consumer_", consumer), [&, consumer] {
          std::vector<int64_t> results;
          Status s = RunConsumer(consumer, starting_rounds[consumer], end_index,
                                 runner, results);
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
