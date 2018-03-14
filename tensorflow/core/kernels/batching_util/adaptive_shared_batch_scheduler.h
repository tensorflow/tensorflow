/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_ADAPTIVE_SHARED_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_ADAPTIVE_SHARED_BATCH_SCHEDULER_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/periodic_function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {
namespace internal {
template <typename TaskType>
class ASBSBatch;

template <typename TaskType>
class ASBSQueue;
}  // namespace internal

// Shared batch scheduler designed to minimize latency. The scheduler keeps
// track of a number of queues (one per model or model version) which are
// continuously enqueuing requests. The scheduler groups the requests into
// batches which it periodically sends off for processing (see
// shared_batch_scheduler.h for more details). AdaptiveSharedBatchScheduler
// (ASBS) prioritizes batches primarily by age (i.e. the batch's oldest request)
// along with a configurable preference for scheduling larger batches first.
//
//
// ASBS tries to keep the system busy by maintaining an adjustable number of
// concurrently processed batches.  If a new batch is created, and the number of
// in flight batches is below the target, the next (i.e. oldest) batch is
// immediately scheduled.  Similarly, when a batch finishes processing, the
// target is rechecked, and another batch may be scheduled.  To avoid the need
// to carefully tune the target for workload, model type, platform, etc, it is
// dynamically adjusted in order to provide the lowest average latency.
//
// Some potential use cases:
// Hardware Accelerators (GPUs & TPUs) - If some phase of batch processing
//   involves serial processing by a device, from a latency perspective it is
//   desirable to keep the device evenly loaded, avoiding the need to wait for
//   the device to process prior batches.
// CPU utilization - If the batch processing is cpu dominated, you can reap
//   latency gains when underutilized by increasing the processing rate, but
//   back the rate off when the load increases to avoid overload.

template <typename TaskType>
class AdaptiveSharedBatchScheduler
    : public std::enable_shared_from_this<
          AdaptiveSharedBatchScheduler<TaskType>> {
 public:
  ~AdaptiveSharedBatchScheduler() {
    // Finish processing batches before destorying other class members.
    batch_thread_pool_.reset();
  }

  struct Options {
    // The name to use for the pool of batch threads.
    string thread_pool_name = {"batch_threads"};
    // Number of batch processing threads - the maximum value of
    // in_flight_batches_limit_.  It is recommended that this value be set by
    // running the system under load, observing the learned value for
    // in_flight_batches_limit_, and setting this maximum to ~ 2x the value.
    // Under low load, in_flight_batches_limit_ has no substantial effect on
    // latency and therefore undergoes a random walk.  Unreasonably large values
    // for num_batch_threads allows for large in_flight_batches_limit_, which
    // will harm latency for some time once load increases again.
    int64 num_batch_threads = port::NumSchedulableCPUs();
    // Although batch selection is primarily based on age, this parameter
    // specifies a preference for larger batches.  A full batch will be
    // scheduled before an older, nearly empty batch as long as the age gap is
    // less than full_batch_scheduling_boost_micros.  The optimal value for this
    // parameter should be of order the batch processing latency, but must be
    // chosen carefully, as too large a value will harm tail latency.
    int64 full_batch_scheduling_boost_micros = 0;
    // The environment to use (typically only overridden by test code).
    Env* env = Env::Default();
    // Initial limit for number of batches being concurrently processed.
    // Non-integer values correspond to probabilistic limits - i.e. a value of
    // 3.2 results in an actual cap of 3 80% of the time, and 4 20% of the time.
    double initial_in_flight_batches_limit = 3;
    // Number of batches between adjustments of in_flight_batches_limit.  Larger
    // numbers will give less noisy latency measurements, but will be less
    // responsive to changes in workload.
    int64 batches_to_average_over = 1000;
  };

  // Ownership is shared between the caller of Create() and any queues created
  // via AddQueue().
  static Status Create(
      const Options& options,
      std::shared_ptr<AdaptiveSharedBatchScheduler<TaskType>>* scheduler);

  struct QueueOptions {
    // Maximum size of each batch.
    int max_batch_size = 1000;
    // Maximum number of enqueued (i.e. non-scheduled) batches.
    int max_enqueued_batches = 10;
  };

  using BatchProcessor = std::function<void(std::unique_ptr<Batch<TaskType>>)>;

  // Adds queue (and its callback) to be managed by this scheduler.
  Status AddQueue(const QueueOptions& options,
                  BatchProcessor process_batch_callback,
                  std::unique_ptr<BatchScheduler<TaskType>>* queue);

  double in_flight_batches_limit() {
    mutex_lock l(mu_);
    return in_flight_batches_limit_;
  }

 private:
  // access to AddBatch, RemoveQueue, GetEnv.
  friend class internal::ASBSQueue<TaskType>;

  explicit AdaptiveSharedBatchScheduler(const Options& options);

  // Tracks processing latency and adjusts in_flight_batches_limit to minimize.
  void CallbackWrapper(const internal::ASBSBatch<TaskType>* batch,
                       BatchProcessor callback);

  // Schedules batch if in_flight_batches_limit_ is not met.
  void MaybeScheduleNextBatch() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Notifies scheduler of non-empty batch which is eligible for processing.
  void AddBatch(const internal::ASBSBatch<TaskType>* batch);

  // Removes queue from scheduler.
  void RemoveQueue(const internal::ASBSQueue<TaskType>* queue);

  Env* GetEnv() const { return options_.env; }

  const Options options_;

  // Collection of batches added by AddBatch, ordered by age. Owned by scheduler
  // until they are released for processing.
  std::vector<const internal::ASBSBatch<TaskType>*> batches_ GUARDED_BY(mu_);

  // Unowned queues and callbacks added by AddQueue.
  std::unordered_map<const internal::ASBSQueue<TaskType>*, BatchProcessor>
      queues_and_callbacks_ GUARDED_BY(mu_);

  mutex mu_;

  // Responsible for running the batch processing callbacks.
  std::unique_ptr<thread::ThreadPool> batch_thread_pool_;

  // Limit on number of batches which can be concurrently processed.
  // Non-integer values correspond to probabilistic limits - i.e. a value of 3.2
  // results in an actual cap of 3 80% of the time, and 4 20% of the time.
  double in_flight_batches_limit_ GUARDED_BY(mu_);

  // Number of batches currently being processed.
  int64 in_flight_batches_ GUARDED_BY(mu_) = 0;

  // RNG engine and distribution.
  std::default_random_engine rand_engine_;
  std::uniform_real_distribution<double> rand_double_;

  // Fields controlling the dynamic adjustment of in_flight_batches_limit_.
  // Number of batches since the last in_flight_batches_limit_ adjustment.
  int64 batch_count_ GUARDED_BY(mu_) = 0;
  // Sum of processing latency for batches counted by batch_count_.
  int64 batch_latency_sum_ GUARDED_BY(mu_) = 0;
  // Average batch latency for previous value of in_flight_batches_limit_.
  double last_avg_latency_ms_ GUARDED_BY(mu_) = 0;
  // Did last_avg_latency_ms_ decrease from the previous last_avg_latency_ms_?
  bool last_latency_decreased_ GUARDED_BY(mu_) = false;
  // Current direction (+-) to adjust in_flight_batches_limit_
  int step_direction_ GUARDED_BY(mu_) = 1;
  // Max adjustment size (as a fraction of in_flight_batches_limit_).
  constexpr static double kMaxStepSizeMultiplier = 0.125;  // 1/8;
  // Min adjustment size (as a fraction of in_flight_batches_limit_).
  constexpr static double kMinStepSizeMultiplier = 0.0078125;  // 1/128
  // Current adjustment size (as a fraction of in_flight_batches_limit_).
  double step_size_multiplier_ GUARDED_BY(mu_) = kMaxStepSizeMultiplier;

  TF_DISALLOW_COPY_AND_ASSIGN(AdaptiveSharedBatchScheduler);
};

//////////////////////////////////////////////////////////
// Implementation details follow. API users need not read.

namespace internal {
// Consolidates tasks into batches, passing them off to the
// AdaptiveSharedBatchScheduler for processing.
template <typename TaskType>
class ASBSQueue : public BatchScheduler<TaskType> {
 public:
  using QueueOptions =
      typename AdaptiveSharedBatchScheduler<TaskType>::QueueOptions;

  ASBSQueue(std::shared_ptr<AdaptiveSharedBatchScheduler<TaskType>> scheduler,
            const QueueOptions& options);

  ~ASBSQueue() override;

  // Adds task to current batch. Fails if the task size is larger than the batch
  // size or if the current batch is full and this queue's number of outstanding
  // batches is at its maximum.
  Status Schedule(std::unique_ptr<TaskType>* task) override;

  // Number of tasks waiting to be scheduled.
  size_t NumEnqueuedTasks() const override;

  // Number of size 1 tasks which could currently be scheduled without failing.
  size_t SchedulingCapacity() const override;

  // Notifies queue that a batch is about to be scheduled; the queue should not
  // place any more tasks in this batch.
  void ReleaseBatch(const ASBSBatch<TaskType>* batch);

  size_t max_task_size() const override { return options_.max_batch_size; }

 private:
  std::shared_ptr<AdaptiveSharedBatchScheduler<TaskType>> scheduler_;
  const QueueOptions options_;
  // Owned by scheduler_.
  ASBSBatch<TaskType>* current_batch_ GUARDED_BY(mu_) = nullptr;
  int64 num_enqueued_batches_ GUARDED_BY(mu_) = 0;
  int64 num_enqueued_tasks_ GUARDED_BY(mu_) = 0;
  mutable mutex mu_;
  TF_DISALLOW_COPY_AND_ASSIGN(ASBSQueue);
};

// Batch which remembers when and by whom it was created.
template <typename TaskType>
class ASBSBatch : public Batch<TaskType> {
 public:
  ASBSBatch(ASBSQueue<TaskType>* queue, int64 creation_time_micros)
      : queue_(queue), creation_time_micros_(creation_time_micros) {}

  ~ASBSBatch() override {}

  ASBSQueue<TaskType>* queue() const { return queue_; }

  int64 creation_time_micros() const { return creation_time_micros_; }

 private:
  ASBSQueue<TaskType>* queue_;
  const int64 creation_time_micros_;
  TF_DISALLOW_COPY_AND_ASSIGN(ASBSBatch);
};
}  // namespace internal

// ---------------- AdaptiveSharedBatchScheduler ----------------

template <typename TaskType>
constexpr double AdaptiveSharedBatchScheduler<TaskType>::kMaxStepSizeMultiplier;

template <typename TaskType>
constexpr double AdaptiveSharedBatchScheduler<TaskType>::kMinStepSizeMultiplier;

template <typename TaskType>
Status AdaptiveSharedBatchScheduler<TaskType>::Create(
    const Options& options,
    std::shared_ptr<AdaptiveSharedBatchScheduler<TaskType>>* scheduler) {
  if (options.num_batch_threads < 1) {
    return errors::InvalidArgument("num_batch_threads must be positive; was ",
                                   options.num_batch_threads);
  }
  if (options.full_batch_scheduling_boost_micros < 0) {
    return errors::InvalidArgument(
        "full_batch_scheduling_boost_micros can't be negative; was ",
        options.full_batch_scheduling_boost_micros);
  }
  if (options.initial_in_flight_batches_limit > options.num_batch_threads) {
    return errors::InvalidArgument(
        "initial_in_flight_batches_limit (",
        options.initial_in_flight_batches_limit,
        ") should not be larger than num_batch_threads (",
        options.num_batch_threads, ")");
  }
  if (options.initial_in_flight_batches_limit < 1) {
    return errors::InvalidArgument(
        "initial_in_flight_batches_limit should be "
        "greater than or equal to 1; was ",
        options.initial_in_flight_batches_limit);
  }
  if (options.batches_to_average_over < 1) {
    return errors::InvalidArgument(
        "batches_to_average_over should be "
        "greater than or equal to 1; was ",
        options.batches_to_average_over);
  }
  scheduler->reset(new AdaptiveSharedBatchScheduler<TaskType>(options));
  return Status::OK();
}

template <typename TaskType>
AdaptiveSharedBatchScheduler<TaskType>::AdaptiveSharedBatchScheduler(
    const Options& options)
    : options_(options),
      in_flight_batches_limit_(options.initial_in_flight_batches_limit),
      rand_double_(0.0, 1.0) {
  std::random_device device;
  rand_engine_.seed(device());
  batch_thread_pool_.reset(new thread::ThreadPool(
      GetEnv(), options.thread_pool_name, options.num_batch_threads));
}

template <typename TaskType>
Status AdaptiveSharedBatchScheduler<TaskType>::AddQueue(
    const QueueOptions& options, BatchProcessor process_batch_callback,
    std::unique_ptr<BatchScheduler<TaskType>>* queue) {
  if (options.max_batch_size <= 0) {
    return errors::InvalidArgument("max_batch_size must be positive; was ",
                                   options.max_batch_size);
  }
  if (options.max_enqueued_batches <= 0) {
    return errors::InvalidArgument(
        "max_enqueued_batches must be positive; was ",
        options.max_enqueued_batches);
  }
  internal::ASBSQueue<TaskType>* asbs_queue_raw;
  queue->reset(asbs_queue_raw = new internal::ASBSQueue<TaskType>(
                   this->shared_from_this(), options));
  mutex_lock l(mu_);
  queues_and_callbacks_[asbs_queue_raw] = process_batch_callback;
  return Status::OK();
}

template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::AddBatch(
    const internal::ASBSBatch<TaskType>* batch) {
  mutex_lock l(mu_);
  batches_.push_back(batch);
  MaybeScheduleNextBatch();
}

template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::RemoveQueue(
    const internal::ASBSQueue<TaskType>* queue) {
  mutex_lock l(mu_);
  queues_and_callbacks_.erase(queue);
}

template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::MaybeScheduleNextBatch() {
  if (batches_.empty() || in_flight_batches_ >= in_flight_batches_limit_)
    return;
  // Non-integer limit handled probabilistially.
  if (in_flight_batches_limit_ - in_flight_batches_ < 1 &&
      rand_double_(rand_engine_) >
          in_flight_batches_limit_ - in_flight_batches_) {
    return;
  }
  auto best_it = batches_.begin();
  double best_score =
      (*best_it)->creation_time_micros() -
      options_.full_batch_scheduling_boost_micros * (*best_it)->size() /
          static_cast<double>((*best_it)->queue()->max_task_size());
  for (auto it = batches_.begin() + 1; it != batches_.end(); it++) {
    const double score =
        (*it)->creation_time_micros() -
        options_.full_batch_scheduling_boost_micros * (*it)->size() /
            static_cast<double>((*it)->queue()->max_task_size());
    if (score < best_score) {
      best_score = score;
      best_it = it;
    }
  }
  const internal::ASBSBatch<TaskType>* batch = *best_it;
  batches_.erase(best_it);
  // Queue may destroy itself after ReleaseBatch is called.
  batch->queue()->ReleaseBatch(batch);
  batch_thread_pool_->Schedule(
      std::bind(&AdaptiveSharedBatchScheduler<TaskType>::CallbackWrapper, this,
                batch, queues_and_callbacks_[batch->queue()]));
  in_flight_batches_++;
}

template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::CallbackWrapper(
    const internal::ASBSBatch<TaskType>* batch,
    AdaptiveSharedBatchScheduler<TaskType>::BatchProcessor callback) {
  int64 start_time = batch->creation_time_micros();
  callback(std::unique_ptr<Batch<TaskType>>(
      const_cast<internal::ASBSBatch<TaskType>*>(batch)));
  int64 end_time = GetEnv()->NowMicros();
  mutex_lock l(mu_);
  in_flight_batches_--;
  batch_count_++;
  batch_latency_sum_ += end_time - start_time;
  // Occasionally adjust in_flight_batches_limit_ to minimize average latency.
  // Although the optimal value may depend on the workload, the latency should
  // be a simple convex function of in_flight_batches_limit_, allowing us to
  // locate the global minimum relatively quickly.
  if (batch_count_ == options_.batches_to_average_over) {
    double current_avg_latency_ms = (batch_latency_sum_ / 1000.) / batch_count_;
    bool current_latency_decreased =
        current_avg_latency_ms < last_avg_latency_ms_;
    if (current_latency_decreased) {
      // If latency improvement was because we're moving in the correct
      // direction, increase step_size so that we can get to the minimum faster.
      // If latency improvement was due to backtracking from a previous failure,
      // decrease step_size in order to refine our location.
      step_size_multiplier_ *= (last_latency_decreased_ ? 2 : 0.5);
      step_size_multiplier_ =
          std::min(step_size_multiplier_, kMaxStepSizeMultiplier);
      step_size_multiplier_ =
          std::max(step_size_multiplier_, kMinStepSizeMultiplier);
    } else {
      // Return (nearly) to previous position and confirm that latency is better
      // there before decreasing step size.
      step_direction_ = -step_direction_;
    }
    in_flight_batches_limit_ +=
        step_direction_ * in_flight_batches_limit_ * step_size_multiplier_;
    in_flight_batches_limit_ =
        std::min(in_flight_batches_limit_,
                 static_cast<double>(options_.num_batch_threads));
    in_flight_batches_limit_ = std::max(in_flight_batches_limit_, 1.0);
    last_avg_latency_ms_ = current_avg_latency_ms;
    last_latency_decreased_ = current_latency_decreased;
    batch_count_ = 0;
    batch_latency_sum_ = 0;
  }
  MaybeScheduleNextBatch();
}

// ---------------- ASBSQueue ----------------

namespace internal {
template <typename TaskType>
ASBSQueue<TaskType>::ASBSQueue(
    std::shared_ptr<AdaptiveSharedBatchScheduler<TaskType>> scheduler,
    const QueueOptions& options)
    : scheduler_(scheduler), options_(options) {}

template <typename TaskType>
ASBSQueue<TaskType>::~ASBSQueue() {
  // Wait until last batch has been scheduled.
  const int kSleepMicros = 1000;
  for (;;) {
    {
      mutex_lock l(mu_);
      if (num_enqueued_batches_ == 0) {
        break;
      }
    }
    scheduler_->GetEnv()->SleepForMicroseconds(kSleepMicros);
  }
  scheduler_->RemoveQueue(this);
}

template <typename TaskType>
Status ASBSQueue<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  ASBSBatch<TaskType>* new_batch = nullptr;
  size_t size = (*task)->size();
  if (size > options_.max_batch_size) {
    return errors::InvalidArgument("Task size ", size,
                                   " is larger than maximum batch size ",
                                   options_.max_batch_size);
  }
  {
    mutex_lock l(mu_);
    // Current batch is full, create another if allowed.
    if (current_batch_ &&
        current_batch_->size() + size > options_.max_batch_size) {
      if (num_enqueued_batches_ >= options_.max_enqueued_batches) {
        return errors::Unavailable("The batch scheduling queue is full");
      }
      current_batch_->Close();
      current_batch_ = nullptr;
    }
    if (!current_batch_) {
      num_enqueued_batches_++;
      current_batch_ = new_batch =
          new ASBSBatch<TaskType>(this, scheduler_->GetEnv()->NowMicros());
    }
    current_batch_->AddTask(std::move(*task));
    num_enqueued_tasks_++;
  }
  // AddBatch must be called outside of lock, since it may call ReleaseBatch.
  if (new_batch != nullptr) scheduler_->AddBatch(new_batch);
  return Status::OK();
}

template <typename TaskType>
void ASBSQueue<TaskType>::ReleaseBatch(const ASBSBatch<TaskType>* batch) {
  mutex_lock l(mu_);
  num_enqueued_batches_--;
  num_enqueued_tasks_ -= batch->num_tasks();
  if (batch == current_batch_) {
    current_batch_->Close();
    current_batch_ = nullptr;
  }
}

template <typename TaskType>
size_t ASBSQueue<TaskType>::NumEnqueuedTasks() const {
  mutex_lock l(mu_);
  return num_enqueued_tasks_;
}

template <typename TaskType>
size_t ASBSQueue<TaskType>::SchedulingCapacity() const {
  mutex_lock l(mu_);
  const int current_batch_capacity =
      current_batch_ ? options_.max_batch_size - current_batch_->size() : 0;
  const int spare_batches =
      options_.max_enqueued_batches - num_enqueued_batches_;
  return spare_batches * options_.max_batch_size + current_batch_capacity;
}
}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_ADAPTIVE_SHARED_BATCH_SCHEDULER_H_
