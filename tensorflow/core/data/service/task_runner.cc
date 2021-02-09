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
// Unavailable error. This prevents the server from hanging on shutdown when
// some round-robin consumers exit earlier than others.
const int64 kTimeoutUs = 60 * 1000 * 1000;  // 1 minute.
// Time to wait before skipping a round if data still isn't available.
const int64 kWaitBeforeSkipUs = 100 * 1000;  // 100ms.

// Interprets `element` as a size-1 vector containing a CompressedElement, and
// moves the element into `resp`. Returns an error if `element` is of unexpected
// size, type, or shape.
Status MoveCompressedElement(std::vector<Tensor>&& element,
                             GetElementResponse& resp) {
  if (element.size() != 1) {
    return errors::FailedPrecondition(
        "Expected dataset to produce a single scalar variant tensor, but the "
        "dataset produced ",
        element.size(), " outputs");
  }
  if (element[0].dtype() != DT_VARIANT) {
    return errors::FailedPrecondition(
        "Expected dataset to produce a single scalar variant tensor, but "
        "the dataset produced a tensor with type ",
        DataTypeString(element[0].dtype()));
  }
  if (!TensorShapeUtils::IsScalar(element[0].shape())) {
    return errors::FailedPrecondition(
        "Expected dataset to produce a single scalar variant tensor, but "
        "the dataset produced a tensor with shape ",
        element[0].shape());
  }
  Variant& variant = element[0].scalar<Variant>()();
  CompressedElement* compressed = variant.get<CompressedElement>();
  if (compressed == nullptr) {
    return errors::FailedPrecondition(
        "Expected dataset to produce a CompressedElement variant tensor, but "
        "it produced ",
        variant.TypeName());
  }
  *resp.mutable_compressed_element() = *compressed;
  return Status::OK();
}
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

Status FirstComeFirstServedTaskRunner::GetNext(const GetElementRequest& req,
                                               GetElementResponse& resp) {
  std::vector<Tensor> element;
  bool end_of_task;
  resp.set_skip_task(false);
  TF_RETURN_IF_ERROR(iterator_->GetNext(element, end_of_task));
  resp.set_end_of_sequence(end_of_task);
  if (!end_of_task) {
    return MoveCompressedElement(std::move(element), resp);
  }
  return Status::OK();
}

RoundRobinTaskRunner::RoundRobinTaskRunner(
    std::unique_ptr<TaskIterator> iterator, int64 num_consumers)
    : num_consumers_(num_consumers),
      buffer_(num_consumers_),
      prefetch_thread_(std::move(iterator), num_consumers_) {
  VLOG(1) << "Creating task runner for distributing data round-robin to "
          << num_consumers << " consumers";
}

Status RoundRobinTaskRunner::ValidateRequest(const GetElementRequest& req) {
  if (req.consumer_index() < 0 || req.round_index() < 0) {
    return errors::FailedPrecondition(
        "RoundRobinTaskRunner needs to know the consumer index and element "
        "index of each request.");
  }
  if (req.consumer_index() >= num_consumers_) {
    return errors::FailedPrecondition(
        "Requesting data for consumer index ", req.consumer_index(),
        ", but the task is configured for only ", num_consumers_, " consumers");
  }
  return Status::OK();
}

Status RoundRobinTaskRunner::PrepareFullRound(int64 wait_us)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(1) << "Preparing full round for index " << current_round_;
  // This was the last request to arrive, time to start a new round.
  TF_RETURN_IF_ERROR(prefetch_thread_.FillBuffer(wait_us, buffer_));
  round_skipped_ = buffer_.empty();
  new_round_cv_.notify_all();
  return Status::OK();
}

Status RoundRobinTaskRunner::PreparePartialRound()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(1) << "Starting partial round for " << requests_[first_round_].size()
          << " consumers";
  current_round_ = first_round_;
  new_round_cv_.notify_all();
  // Indicates that we need a partial round to get consumers back in sync.
  auto next_round_request = *(requests_[first_round_ + 1].begin());
  if (next_round_request->skipped_previous_round()) {
    VLOG(1) << "Skipping partial round";
    round_skipped_ = true;
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(prefetch_thread_.FillBuffer(/*wait_us=*/-1, buffer_));
  round_skipped_ = false;
  return Status::OK();
}

Status RoundRobinTaskRunner::PrepareRound(const GetElementRequest& req) {
  mutex_lock l(mu_);
  absl::flat_hash_set<const GetElementRequest*>& round =
      requests_[req.round_index()];
  first_round_ = std::min(first_round_, req.round_index());
  round.insert(&req);
  if (current_round_ < req.round_index() && round.size() == num_consumers_) {
    current_round_ = req.round_index();
    int64 wait_us = kWaitBeforeSkipUs;
    if (!req.allow_skip()) {
      wait_us = -1;
    }
    TF_RETURN_IF_ERROR(PrepareFullRound(wait_us));
  }
  if (current_round_ < 0 &&
      requests_[first_round_].size() + requests_[first_round_ + 1].size() ==
          num_consumers_) {
    TF_RETURN_IF_ERROR(PreparePartialRound());
  }
  while (current_round_ < req.round_index()) {
    TF_RETURN_IF_ERROR(prefetch_thread_.GetStatus());
    std::cv_status s =
        new_round_cv_.wait_for(l, std::chrono::microseconds(kTimeoutUs));
    if (s == std::cv_status::timeout) {
      // Clients will retry Unavailable.
      return errors::Unavailable(
          "Timeout waiting for other round-robin consumers to be ready.");
    }
  }
  return prefetch_thread_.GetStatus();
}

Status RoundRobinTaskRunner::GetNext(const GetElementRequest& req,
                                     GetElementResponse& resp) {
  TF_RETURN_IF_ERROR(ValidateRequest(req));
  resp.set_end_of_sequence(false);
  VLOG(2) << "Received request from consumer index " << req.consumer_index()
          << " for round " << req.round_index();
  TF_RETURN_IF_ERROR(PrepareRound(req));
  tf_shared_lock l(mu_);
  resp.set_skip_task(round_skipped_);
  if (round_skipped_) {
    VLOG(1) << "Buffer not ready, skipping round " << current_round_
            << " for consumer " << req.consumer_index();
    return Status::OK();
  }
  std::vector<Tensor> element;
  for (auto& component : buffer_[req.consumer_index()]) {
    element.push_back(tensor::DeepCopy(component));
  }
  if (VLOG_IS_ON(2)) {
    int64 size = 0;
    for (auto& component : element) {
      size += component.TotalBytes();
    }
    VLOG(2) << "Returning to consumer " << req.consumer_index() << " for round "
            << req.round_index() << ". element size " << size;
  }
  return MoveCompressedElement(std::move(element), resp);
}

PrefetchThread::PrefetchThread(std::unique_ptr<TaskIterator> iterator,
                               int64 round_size)
    : iterator_(std::move(iterator)), round_size_(round_size) {
  thread_ = absl::WrapUnique(
      Env::Default()->StartThread({}, "round-robin-prefetch", [&] { Run(); }));
}

PrefetchThread::~PrefetchThread() {
  mutex_lock l(mu_);
  cancelled_ = true;
  cv_.notify_all();
}

void PrefetchThread::Run() {
  while (true) {
    {
      mutex_lock l(mu_);
      while (!cancelled_ && buffer_.size() >= round_size_) {
        cv_.wait(l);
      }
      if (cancelled_) {
        return;
      }
    }
    std::vector<Tensor> element;
    bool end_of_sequence;
    Status s = iterator_->GetNext(element, end_of_sequence);
    if (!s.ok()) {
      mutex_lock l(mu_);
      status_ = s;
      cv_.notify_all();
      return;
    }
    if (end_of_sequence) {
      mutex_lock l(mu_);
      status_ = errors::FailedPrecondition(
          "Encountered end of sequence on a round-robin read iterator. "
          "Please ensure that the dataset used for round-robin reading has "
          "infinite cardinality, e.g. by adding a .repeat() transformation "
          "at the end.");
      cv_.notify_all();
      return;
    }
    mutex_lock l(mu_);
    buffer_.push_back(std::move(element));
    cv_.notify_all();
  }
}

Status PrefetchThread::FillBuffer(int64 wait_us,
                                  std::vector<std::vector<Tensor>>& out) {
  int64 start_us = Env::Default()->NowMicros();
  out.clear();
  mutex_lock l(mu_);
  while (buffer_.size() < round_size_ && !cancelled_ && status_.ok()) {
    int64 remaining_us = start_us + wait_us - Env::Default()->NowMicros();
    if (wait_us >= 0 && remaining_us <= 0) {
      break;
    }
    cv_.wait_for(l, std::chrono::microseconds(remaining_us));
  }
  TF_RETURN_IF_ERROR(status_);
  if (cancelled_) {
    return errors::Cancelled("Prefetch thread cancelled");
  }
  if (buffer_.size() < round_size_) {
    DCHECK_GE(wait_us, 0);
    return Status::OK();
  }
  for (auto& elem : buffer_) {
    out.push_back(std::move(elem));
  }
  buffer_.clear();
  cv_.notify_all();
  return Status::OK();
}

Status PrefetchThread::GetStatus() {
  mutex_lock l(mu_);
  return status_;
}
}  // namespace data
}  // namespace tensorflow
