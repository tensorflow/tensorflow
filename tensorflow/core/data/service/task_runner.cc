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
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/cross_trainer_cache.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/logging_utils.h"
#include "tensorflow/core/data/service/thread_safe_buffer.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {
namespace {
// Time to wait before skipping a round if data still isn't available.
constexpr int64_t kWaitBeforeSkipUs = 100 * 1000;  // 100ms.
constexpr size_t kDefaultCrossTrainerCacheSizeBytes =
    10 * (size_t{1} << 30);  // 10GB

}  // namespace

StandaloneTaskIterator::StandaloneTaskIterator(
    std::unique_ptr<standalone::Dataset> dataset,
    std::unique_ptr<standalone::Iterator> iterator)
    : dataset_(std::move(dataset)), iterator_(std::move(iterator)) {}

Status StandaloneTaskIterator::GetNext(std::vector<Tensor>& element,
                                       bool& end_of_sequence) {
  return iterator_->GetNext(&element, &end_of_sequence);
}

int64_t StandaloneTaskIterator::Cardinality() const {
  return dataset_->Get()->Cardinality();
}

StatusOr<std::vector<Tensor>> StandaloneTaskIterator::Save() {
  return iterator_->Save();
}

Status StandaloneTaskIterator::Restore(
    const std::vector<Tensor>& saved_iterator) {
  return iterator_->Restore(saved_iterator);
}

Status TaskRunner::Create(const experimental::WorkerConfig& worker_config,
                          const TaskDef& task_def,
                          std::unique_ptr<TaskIterator> iterator,
                          std::unique_ptr<TaskRunner>& out) {
  if (task_def.optional_num_consumers_case() == TaskDef::kNumConsumers) {
    int64_t cardinality = iterator->Cardinality();
    if (cardinality != kInfiniteCardinality &&
        cardinality != kUnknownCardinality) {
      return errors::FailedPrecondition(
          "Round robin reads require that the input dataset has infinite "
          "cardinality, but the dataset has cardinality ",
          cardinality,
          ". Consider adding a `.repeat()` transformation to the dataset.");
    }
    out = std::make_unique<RoundRobinTaskRunner>(std::move(iterator),
                                                 task_def.num_consumers(),
                                                 task_def.worker_address());
  } else if (task_def.use_cross_trainer_cache()) {
    const size_t max_cache_size_bytes =
        worker_config.cross_trainer_cache_size_bytes() > 0
            ? worker_config.cross_trainer_cache_size_bytes()
            : kDefaultCrossTrainerCacheSizeBytes;
    out = std::make_unique<CachingTaskRunner>(std::move(iterator),
                                              max_cache_size_bytes);
  } else {
    out = std::make_unique<FirstComeFirstServedTaskRunner>(std::move(iterator));
  }
  return OkStatus();
}

FirstComeFirstServedTaskRunner::FirstComeFirstServedTaskRunner(
    std::unique_ptr<TaskIterator> iterator)
    : iterator_(std::move(iterator)), buffer_(/*buffer_size=*/1) {
  RunPrefetchThread();
}

FirstComeFirstServedTaskRunner::~FirstComeFirstServedTaskRunner() { Cancel(); }

Status FirstComeFirstServedTaskRunner::GetNext(const GetElementRequest& req,
                                               GetElementResult& result) {
  return GetNext(result);
}

Status FirstComeFirstServedTaskRunner::GetNext(GetElementResult& result) {
  TF_ASSIGN_OR_RETURN(result, buffer_.Pop());
  return OkStatus();
}

Status FirstComeFirstServedTaskRunner::PrefetchFn() {
  while (true) {
    TF_RETURN_IF_ERROR(buffer_.Push(GetNextFromInputIterator()));
  }
  return OkStatus();
}

void FirstComeFirstServedTaskRunner::RunPrefetchThread() {
  auto prefetch_fn = [this] {
    Status status = PrefetchFn();
    if (!status.ok()) {
      buffer_.Cancel(status);
    }
  };
  prefetch_thread_ = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"tf_data_service_fcfs_prefetch_thread",
      prefetch_fn));
}

StatusOr<GetElementResult>
FirstComeFirstServedTaskRunner::GetNextFromInputIterator()
    TF_LOCKS_EXCLUDED(mu_) {
  GetElementResult result;
  std::vector<Tensor> element;
  bool end_of_task = false;
  result.skip = false;
  {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(iterator_->GetNext(element, end_of_task));
    result.end_of_sequence = end_of_task;
    result.element_index = element_index_++;
  }
  if (!end_of_task) {
    result.components = std::move(element);
  }
  return result;
}

void FirstComeFirstServedTaskRunner::Cancel() {
  VLOG(2) << "Cancelling tf.data service FCFS task.";
  buffer_.Cancel(errors::Cancelled("tf.data service FCFS task is cancelled."));
}

CachingTaskRunner::CachingTaskRunner(std::unique_ptr<TaskIterator> iterator,
                                     size_t max_cache_size_bytes)
    : fcfs_task_runner_(std::move(iterator)),
      cache_(max_cache_size_bytes,
             std::make_unique<GetElementResultSequence>(fcfs_task_runner_)) {
  LOG(INFO) << "Initialized tf.data service cross-trainer cache with "
            << FormatBytes(max_cache_size_bytes) << " of memory.";
}

CachingTaskRunner::~CachingTaskRunner() { Cancel(); }

Status CachingTaskRunner::GetNext(const GetElementRequest& req,
                                  GetElementResult& result) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<const GetElementResult> element,
                      cache_.Get(req.trainer_id()));
  result = element->Copy();
  return OkStatus();
}

CachingTaskRunner::GetElementResultSequence::GetElementResultSequence(
    FirstComeFirstServedTaskRunner& fcfs_task_runner)
    : fcfs_task_runner_(fcfs_task_runner) {}

StatusOr<GetElementResult>
CachingTaskRunner::GetElementResultSequence::GetNext() {
  GetElementResult result;
  TF_RETURN_IF_ERROR(fcfs_task_runner_.GetNext(result));
  if (result.end_of_sequence) {
    return errors::InvalidArgument(
        "Cross-trainer caching requires the input dataset to be infinite. "
        "However, it reached the end of sequence.");
  }
  return result;
}

size_t CachingTaskRunner::GetElementResultSequence::GetElementSizeBytes(
    const GetElementResult& element) const {
  return element.EstimatedMemoryUsageBytes();
}

void CachingTaskRunner::Cancel() {
  VLOG(2) << "Cancelling tf.data service cross-trainer cache task.";
  if (!cache_.IsCancelled()) {
    cache_.Cancel(errors::Cancelled(
        "tf.data service cross-trainer cache task is cancelled."));
  }
  fcfs_task_runner_.Cancel();
}

RoundRobinTaskRunner::RoundRobinTaskRunner(
    std::unique_ptr<TaskIterator> iterator, int64_t num_consumers,
    string worker_address)
    : num_consumers_(num_consumers),
      worker_address_(worker_address),
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
  return OkStatus();
}

Status RoundRobinTaskRunner::PrepareFullRound(int64_t wait_us)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(1) << worker_address_ << ": Preparing full round for round "
          << current_round_;
  // This was the last request to arrive, time to start a new round.
  TF_RETURN_IF_ERROR(prefetch_thread_.FillBuffer(wait_us, buffer_));
  round_skipped_ = buffer_.empty();
  new_round_cv_.notify_all();
  return OkStatus();
}

Status RoundRobinTaskRunner::PreparePartialRound()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(1) << worker_address_ << ": Starting partial round " << first_round_
          << " for " << requests_[first_round_].size() << " consumers";
  current_round_ = first_round_;
  new_round_cv_.notify_all();
  // Indicates that we need a partial round to get consumers back in sync.
  auto next_round_request = *(requests_[first_round_ + 1].begin()->second);
  if (next_round_request.skipped_previous_round()) {
    VLOG(1) << "Skipping partial round";
    round_skipped_ = true;
    return OkStatus();
  }
  TF_RETURN_IF_ERROR(prefetch_thread_.FillBuffer(/*wait_us=*/-1, buffer_));
  round_skipped_ = false;
  return OkStatus();
}

Status RoundRobinTaskRunner::PrepareRound(const GetElementRequest& req) {
  mutex_lock l(mu_);
  first_round_ = std::min(first_round_, req.round_index());
  absl::flat_hash_map<int64_t, const GetElementRequest*>& round =
      requests_[req.round_index()];
  round[req.consumer_index()] = &req;
  auto cleanup = gtl::MakeCleanup([&]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    requests_[req.round_index()].erase(req.consumer_index());
  });
  if (current_round_ < req.round_index() && round.size() == num_consumers_) {
    current_round_ = req.round_index();
    int64_t wait_us = kWaitBeforeSkipUs;
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
  while (!cancelled_ && current_round_ < req.round_index()) {
    TF_RETURN_IF_ERROR(prefetch_thread_.GetStatus());
    new_round_cv_.wait(l);
  }
  if (current_round_ < req.round_index() && cancelled_) {
    return errors::Cancelled("Worker is shutting down.");
  }
  if (current_round_ != req.round_index()) {
    return errors::FailedPrecondition(
        "Consumer ", req.consumer_index(), " requested data for round ",
        req.round_index(), ", but the current round has already reached ",
        current_round_,
        ". This may indicate that the consumer was restarted with the same "
        "iteration "
        "name.`");
  }
  return prefetch_thread_.GetStatus();
}

Status RoundRobinTaskRunner::GetNext(const GetElementRequest& req,
                                     GetElementResult& result) {
  TF_RETURN_IF_ERROR(ValidateRequest(req));
  result.end_of_sequence = false;
  VLOG(2) << worker_address_ << ": Received request from consumer index "
          << req.consumer_index() << " for round " << req.round_index();
  TF_RETURN_IF_ERROR(PrepareRound(req));
  tf_shared_lock l(mu_);
  result.skip = round_skipped_;
  if (round_skipped_) {
    VLOG(1) << worker_address_ << ": Buffer not ready, skipping round "
            << current_round_ << " for consumer " << req.consumer_index();
    return OkStatus();
  }
  auto& buffer_result = buffer_[req.consumer_index()];
  result.element_index = buffer_result->index;
  std::vector<Tensor> element;
  for (auto& component : buffer_result->components) {
    element.push_back(tensor::DeepCopy(component));
  }
  if (VLOG_IS_ON(2)) {
    int64_t size = 0;
    for (auto& component : element) {
      size += component.TotalBytes();
    }
    VLOG(2) << worker_address_ << ": Returning element " << result.element_index
            << " to consumer " << req.consumer_index() << " for round "
            << req.round_index() << ". element size " << size;
  }
  result.components = std::move(element);
  return OkStatus();
}

void RoundRobinTaskRunner::Cancel() {
  mutex_lock l(mu_);
  cancelled_ = true;
  new_round_cv_.notify_all();
}

PrefetchThread::PrefetchThread(std::unique_ptr<TaskIterator> iterator,
                               int64_t round_size)
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
    buffer_.push_back(std::make_unique<Element>(std::move(element), index_++));
    cv_.notify_all();
  }
}

Status PrefetchThread::FillBuffer(int64_t wait_us,
                                  std::vector<std::unique_ptr<Element>>& out) {
  int64_t start_us = Env::Default()->NowMicros();
  out.clear();
  mutex_lock l(mu_);
  while (buffer_.size() < round_size_ && !cancelled_ && status_.ok()) {
    int64_t remaining_us = start_us + wait_us - Env::Default()->NowMicros();
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
    return OkStatus();
  }
  for (auto& elem : buffer_) {
    out.push_back(std::move(elem));
  }
  buffer_.clear();
  cv_.notify_all();
  return OkStatus();
}

Status PrefetchThread::GetStatus() {
  mutex_lock l(mu_);
  return status_;
}
}  // namespace data
}  // namespace tensorflow
