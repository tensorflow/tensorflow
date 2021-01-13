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

#include "tensorflow/core/data/compression_utils.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace {
// How long to wait for other round-robin consumers before returning with an
// Unavailable error. The unavailable error gives the client an opportunity to
// either give up or retry to continue waiting.
const int64 kDefaultTimeoutUs = 2 * 1000 * 1000;  // 2 seconds.
}  // namespace

StandaloneTaskIterator::StandaloneTaskIterator(
    std::unique_ptr<standalone::Dataset> dataset,
    std::unique_ptr<standalone::Iterator> iterator)
    : dataset_(std::move(dataset)), iterator_(std::move(iterator)) {}

Status StandaloneTaskIterator::GetNext(std::vector<Tensor>& element,
                                       bool& end_of_sequence) {
  return iterator_->GetNext(&element, &end_of_sequence);
}

int64 StandaloneTaskIterator::Cardinality() const {
  return dataset_->Get()->Cardinality();
}

Status TaskRunner::Create(const TaskDef& task_def,
                          std::unique_ptr<TaskIterator> iterator,
                          std::unique_ptr<TaskRunner>& out) {
  if (task_def.optional_num_consumers_case() == TaskDef::kNumConsumers) {
    int64 cardinality = iterator->Cardinality();
    if (cardinality != kInfiniteCardinality &&
        cardinality != kUnknownCardinality) {
      return errors::FailedPrecondition(
          "Round robin reads require that the input dataset has infinite "
          "cardinality, but the dataset has cardinality ",
          cardinality,
          ". Consider adding a `.repeat()` transformation to the dataset.");
    }
    out = absl::make_unique<RoundRobinTaskRunner>(
        std::move(iterator), task_def.num_consumers(), kDefaultTimeoutUs);
  } else {
    out =
        absl::make_unique<FirstComeFirstServedTaskRunner>(std::move(iterator));
  }
  return Status::OK();
}

FirstComeFirstServedTaskRunner::FirstComeFirstServedTaskRunner(
    std::unique_ptr<TaskIterator> iterator)
    : iterator_(std::move(iterator)) {}

Status FirstComeFirstServedTaskRunner::GetNext(const Request& request,
                                               std::vector<Tensor>& element,
                                               bool& end_of_task) {
  return iterator_->GetNext(element, end_of_task);
}

RoundRobinTaskRunner::RoundRobinTaskRunner(
    std::unique_ptr<TaskIterator> iterator, int64 num_consumers,
    int64 timeout_us)
    : num_consumers_(num_consumers),
      timeout_us_(timeout_us),
      iterator_(std::move(iterator)),
      buffer_(num_consumers_) {
  VLOG(1) << "Creating task runner for distributing data round-robin to "
          << num_consumers << " consumers";
}

Status RoundRobinTaskRunner::GetNext(const Request& request,
                                     std::vector<Tensor>& element,
                                     bool& end_of_task) {
  if (request.consumer_index < 0 || request.round_index < 0) {
    return errors::FailedPrecondition(
        "RoundRobinTaskRunner needs to know the consumer index and element "
        "index of each request.");
  }
  if (request.consumer_index >= num_consumers_) {
    return errors::FailedPrecondition(
        "Requesting data for consumer index ", request.consumer_index,
        ", but the task is configured for only ", num_consumers_, " consumers");
  }
  VLOG(2) << "Received request from consumer index " << request.consumer_index
          << " for round " << request.round_index;

  mutex_lock l(mu_);
  absl::flat_hash_set<int64>& round = requests_[request.round_index];
  first_round_ = std::min(first_round_, request.round_index);
  round.insert(request.consumer_index);
  if (current_round_ < request.round_index && round.size() == num_consumers_) {
    VLOG(1) << "Starting normal round with round index " << request.round_index;
    // This was the last request to arrive, time to start a new round.
    TF_RETURN_IF_ERROR(FillBuffer());
    current_round_ = request.round_index;
    new_round_cv_.notify_all();
  }
  if (current_round_ < 0 &&
      requests_[first_round_].size() + requests_[first_round_ + 1].size() ==
          num_consumers_) {
    VLOG(1) << "Starting partial round for " << requests_[first_round_].size()
            << " consumers";
    // Indicates that we need a partial round to get consumers back in sync.
    TF_RETURN_IF_ERROR(FillBuffer());
    current_round_ = first_round_;
    new_round_cv_.notify_all();
  }
  while (current_round_ < request.round_index) {
    std::cv_status s =
        new_round_cv_.wait_for(l, std::chrono::microseconds(timeout_us_));
    if (s == std::cv_status::timeout) {
      // Clients will retry Unavailable.
      return errors::Unavailable(
          "Timeout waiting for other round-robin consumers to be ready.");
    }
  }
  end_of_task = end_of_task_;
  if (!end_of_task) {
    element.clear();
    for (auto& component : buffer_[request.consumer_index]) {
      element.push_back(tensor::DeepCopy(component));
    }
  }
  VLOG(2) << "Returning to consumer " << request.consumer_index << " for round "
          << request.round_index;
  return Status::OK();
}

Status RoundRobinTaskRunner::FillBuffer() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  for (int i = 0; i < num_consumers_; ++i) {
    buffer_[i].clear();
    bool end_of_sequence;
    TF_RETURN_IF_ERROR(iterator_->GetNext(buffer_[i], end_of_sequence));
    if (end_of_sequence) {
      return errors::FailedPrecondition(
          "Encountered end of sequence on a round-robin read iterator. Please "
          "ensure that the dataset used for round-robin reading has infinite "
          "cardinality, e.g. by adding a .repeat() transformation at the end.");
    }
  }
  return Status::OK();
}
}  // namespace data
}  // namespace tensorflow
