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

#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {

StandaloneTaskIterator::StandaloneTaskIterator(
    std::unique_ptr<standalone::Dataset> dataset,
    std::unique_ptr<standalone::Iterator> iterator)
    : dataset_(std::move(dataset)), iterator_(std::move(iterator)) {}

Status StandaloneTaskIterator::GetNext(std::vector<Tensor>& element,
                                       bool& end_of_sequence) {
  return iterator_->GetNext(&element, &end_of_sequence);
}

Status TaskRunner::Create(const TaskDef& task_def,
                          std::unique_ptr<TaskIterator> iterator,
                          std::unique_ptr<TaskRunner>& out) {
  if (task_def.optional_num_consumers_case() == TaskDef::kNumConsumers) {
    out = absl::make_unique<RoundRobinTaskRunner>(std::move(iterator),
                                                  task_def.num_consumers());
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
    std::unique_ptr<TaskIterator> iterator, int64 num_consumers)
    : num_consumers_(num_consumers),
      iterator_(std::move(iterator)),
      buffer_(num_consumers_) {}

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

  mutex_lock l(mu_);
  absl::flat_hash_set<int64>& round = requests_[request.round_index];
  first_round_ = std::min(first_round_, request.round_index);
  round.insert(request.consumer_index);
  if (current_round_ < request.round_index && round.size() == num_consumers_) {
    // This was the last request to arrive, time to start a new round.
    TF_RETURN_IF_ERROR(FillBuffer());
    current_round_ = request.round_index;
    new_round_cv_.notify_all();
  }
  if (current_round_ < 0 &&
      requests_[first_round_].size() + requests_[first_round_ + 1].size() ==
          num_consumers_) {
    // Indicates that we need a partial round to get consumers back in sync.
    TF_RETURN_IF_ERROR(FillBuffer());
    current_round_ = first_round_;
    new_round_cv_.notify_all();
  }
  while (current_round_ < request.round_index) {
    new_round_cv_.wait(l);
  }
  Result& result = buffer_[request.consumer_index];
  end_of_task = result.end_of_task;
  if (!end_of_task) {
    element = std::move(result.element);
  }
  return Status::OK();
}

Status RoundRobinTaskRunner::FillBuffer() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  for (int i = 0; i < num_consumers_; ++i) {
    Result& result = buffer_[i];
    result.element.clear();
    TF_RETURN_IF_ERROR(iterator_->GetNext(result.element, result.end_of_task));
    if (buffer_[i].end_of_task && !buffer_[0].end_of_task) {
      std::vector<Tensor>& first_element = buffer_[0].element;
      // Pad out the round with empty elements.
      buffer_[i].element.clear();
      for (int c = 0; c < first_element.size(); ++c) {
        TensorShape shape = first_element[c].shape();
        if (shape.dims() > 0) {
          shape.set_dim(0, 0);
        }
        buffer_[i].element.push_back(Tensor(first_element[c].dtype(), shape));
      }
      buffer_[i].end_of_task = false;
    }
  }
  return Status::OK();
}
}  // namespace data
}  // namespace tensorflow
