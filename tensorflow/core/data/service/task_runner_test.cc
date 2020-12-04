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

#include "absl/memory/memory.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {
class TestTaskIterator : public TaskIterator {
 public:
  explicit TestTaskIterator(const std::vector<std::vector<Tensor>>& elements)
      : elements_(elements), index_(0) {}

  Status GetNext(std::vector<Tensor>& element, bool& end_of_sequence) override {
    end_of_sequence = index_ >= elements_.size();
    if (!end_of_sequence) {
      element = elements_[index_++];
    }
    return Status::OK();
  }

 private:
  std::vector<std::vector<Tensor>> elements_;
  int64 index_;
};

// Reads from the task runner, storing results in `*output`.
Status RunConsumer(int64 consumer_index, int64 start_index,
                   TaskRunner& task_runner,
                   std::vector<std::vector<Tensor>>& output) {
  bool end_of_sequence = false;
  int64 next_index = start_index;
  while (!end_of_sequence) {
    TaskRunner::Request request;
    request.round_index = next_index++;
    request.consumer_index = consumer_index;
    std::vector<Tensor> element;
    TF_RETURN_IF_ERROR(task_runner.GetNext(request, element, end_of_sequence));
    if (!end_of_sequence) {
      output.push_back(element);
    }
  }
  return Status::OK();
}
}  // namespace

TEST(FirstComeFirstServedTaskRunner, GetNext) {
  std::vector<std::vector<Tensor>> elements;
  for (int64 i = 0; i < 10; ++i) {
    std::vector<Tensor> element;
    element.push_back(Tensor(i));
    elements.push_back(element);
  }
  FirstComeFirstServedTaskRunner runner(
      absl::make_unique<TestTaskIterator>(elements));
  TaskRunner::Request request;
  for (auto& expected_element : elements) {
    std::vector<Tensor> element;
    bool end_of_sequence;
    TF_ASSERT_OK(runner.GetNext(request, element, end_of_sequence));
    ASSERT_FALSE(end_of_sequence);
    ASSERT_EQ(element.size(), 1);
    test::ExpectEqual(element[0], expected_element[0]);
  }
  for (int i = 0; i < 2; ++i) {
    std::vector<Tensor> element;
    bool end_of_sequence;
    TF_ASSERT_OK(runner.GetNext(request, element, end_of_sequence));
    ASSERT_TRUE(end_of_sequence);
  }
}

class ConsumeParallelTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<int64, int64>> {};

TEST_P(ConsumeParallelTest, ConsumeParallel) {
  int64 num_elements = std::get<0>(GetParam());
  int64 num_consumers = std::get<1>(GetParam());
  std::vector<std::vector<Tensor>> elements;
  for (int64 i = 0; i < num_elements; ++i) {
    std::vector<Tensor> element;
    element.push_back(Tensor(i));
    elements.push_back(element);
  }
  RoundRobinTaskRunner runner(absl::make_unique<TestTaskIterator>(elements),
                              num_consumers);
  std::vector<std::vector<std::vector<Tensor>>> per_consumer_results;
  std::vector<std::unique_ptr<Thread>> consumers;
  mutex mu;
  Status error;
  for (int consumer = 0; consumer < num_consumers; ++consumer) {
    mutex_lock l(mu);
    per_consumer_results.emplace_back();
    consumers.push_back(absl::WrapUnique(Env::Default()->StartThread(
        {}, absl::StrCat("consumer_", consumer), [&, consumer] {
          std::vector<std::vector<Tensor>> results;
          Status s = RunConsumer(consumer, /*start_index=*/0, runner, results);
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
    Tensor expected = elements[i][0];
    test::ExpectEqual(per_consumer_results[consumer][round][0], expected);
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
  int64 num_elements = 20;
  int64 num_consumers = 5;
  std::vector<int64> starting_rounds = {12, 11, 11, 12, 12};
  int64 min_starting_round = 11;
  std::vector<std::vector<Tensor>> elements;
  for (int64 i = 0; i < num_elements; ++i) {
    std::vector<Tensor> element;
    element.push_back(Tensor(i));
    elements.push_back(element);
  }
  RoundRobinTaskRunner runner(absl::make_unique<TestTaskIterator>(elements),
                              num_consumers);
  std::vector<std::vector<std::vector<Tensor>>> per_consumer_results;
  std::vector<std::unique_ptr<Thread>> consumers;
  mutex mu;
  Status error;
  for (int consumer = 0; consumer < num_consumers; ++consumer) {
    mutex_lock l(mu);
    per_consumer_results.emplace_back();
    consumers.push_back(absl::WrapUnique(Env::Default()->StartThread(
        {}, absl::StrCat("consumer_", consumer), [&, consumer] {
          std::vector<std::vector<Tensor>> results;
          Status s =
              RunConsumer(consumer, starting_rounds[consumer], runner, results);
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
    auto& results = per_consumer_results[consumer];
    int start = consumer;
    int expected_elements = num_elements / num_consumers;
    if (starting_rounds[consumer] != min_starting_round) {
      start += num_consumers;
      expected_elements--;
    }
    ASSERT_EQ(results.size(), expected_elements);
    int index = 0;
    for (int i = start; i < num_elements; i += num_consumers) {
      Tensor expected = elements[i][0];
      test::ExpectEqual(results[index++][0], expected);
    }
  }
}
}  // namespace data
}  // namespace tensorflow
