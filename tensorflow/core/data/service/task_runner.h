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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_TASK_RUNNER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_TASK_RUNNER_H_

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace data {

// Iterator over a task's elements.
class TaskIterator {
 public:
  virtual ~TaskIterator() = default;
  // If the iterator is not yet exhausted, `GetNext` stores the next element in
  // `element` and sets `end_of_sequence` to `false`. Otherwise, sets
  // `end_of_sequence to `true`.
  virtual Status GetNext(std::vector<Tensor>& element,
                         bool& end_of_sequence) = 0;
};

// Implementation of TaskIterator wrapping a standalone iterator.
class StandaloneTaskIterator : public TaskIterator {
 public:
  // `dataset` should be the dataset that created `iterator`.
  // StandaloneTaskIterator takes ownership of the dataset to ensures it
  // lives as long as `iterator`.
  StandaloneTaskIterator(std::unique_ptr<standalone::Dataset> dataset,
                         std::unique_ptr<standalone::Iterator> iterator);
  Status GetNext(std::vector<Tensor>& element, bool& end_of_sequence) override;

 private:
  std::unique_ptr<standalone::Dataset> dataset_;
  std::unique_ptr<standalone::Iterator> iterator_;
};

// Interface for providing elements to task consumers.
class TaskRunner {
 public:
  struct Request {
    // Optional consumer index indicating which consumer is making the request.
    // Only needed for round-robin reads.
    int64 consumer_index = -1;
    // Optional round index indicating which round the consumer wants to read
    // from. Consumers are expected to read from consecutive rounds, starting
    // with round 0. The task runner will attempt to serve all consumer
    // requests for a round from the same block of `num_consumers` iterator
    // indices, where block `n` is defined as elements `n*num_consumers` to
    // `(n+1)*num_consumers`.
    int64 round_index = -1;
  };

  // Creates a `TaskRunner` and stores it in `out`.
  static Status Create(const TaskDef& task_def,
                       std::unique_ptr<TaskIterator> iterator,
                       std::unique_ptr<TaskRunner>& out);
  virtual ~TaskRunner() = default;
  // Gets the next element for the given request, storing the results in
  // `element` and `end_of_task`.
  virtual Status GetNext(const Request& request, std::vector<Tensor>& element,
                         bool& end_of_task) = 0;
};

// A task runner which provides elements on a first-come first-served basis.
// It does not consider which consumer is making the request.
class FirstComeFirstServedTaskRunner : public TaskRunner {
 public:
  explicit FirstComeFirstServedTaskRunner(
      std::unique_ptr<TaskIterator> iterator);
  Status GetNext(const Request& request, std::vector<Tensor>& element,
                 bool& end_of_task) override;

 private:
  std::unique_ptr<TaskIterator> iterator_;
};

// A task runner which enforces round-robin order for consuming a task's
// elements. Requests must provide a consumer index and element index.
// `RoundRobinTaskRunner` provides elements in a series of "rounds". In each
// successive round, the runner waits to receive requests from all consumers.
// These requests are blocked until all requests arrive. Once all requests
// arrive, the runner hands out elements to consumers in order of their consumer
// indices.
//
// Consumers are expected to successively request consecutive element indices,
// starting at 0. The same element can be requested multiple times by the same
// consumer, as long as the consumer hasn't yet requested the next element (at
// the start of each round we discard elements from the previous round).
//
// If the worker restarts mid-round, a situation arises where some consumers
// are requesting element index `n` while others are requesting element index
// `n + 1`. To remedy this, the first round after restart may be a partial
// round, where we only serve elements to consumers requesting data for element
// index `n`, blocking other consumers until the second round.
class RoundRobinTaskRunner : public TaskRunner {
 public:
  RoundRobinTaskRunner(std::unique_ptr<TaskIterator> iterator,
                       int64 num_consumers);
  Status GetNext(const Request& request, std::vector<Tensor>& element,
                 bool& end_of_task) override;

 private:
  struct Result {
    std::vector<Tensor> element;
    bool end_of_task = false;
  };
  // Fills `buffer_` with `num_consumers_` elements.
  Status FillBuffer();

  const int64 num_consumers_;
  std::unique_ptr<TaskIterator> iterator_;
  mutex mu_;
  // Condition variable notified whenever we start a new round of round-robin.
  condition_variable new_round_cv_;
  // Map from round number to consumers waiting for data from that round.
  absl::flat_hash_map<int64, absl::flat_hash_set<int64>> requests_
      TF_GUARDED_BY(mu_);
  // Index of the first round we plan to serve. At startup, this is the minimum
  // of all requested element indices.
  int64 first_round_ TF_GUARDED_BY(mu_) = kint64max;
  int64 current_round_ TF_GUARDED_BY(mu_) = -1;
  // Buffered results for the current round.
  std::vector<Result> buffer_ TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_TASK_RUNNER_H_
