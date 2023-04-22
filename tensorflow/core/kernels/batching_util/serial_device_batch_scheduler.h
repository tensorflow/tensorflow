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
#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_SERIAL_DEVICE_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_SERIAL_DEVICE_BATCH_SCHEDULER_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
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
class SDBSBatch;

template <typename TaskType>
class SDBSQueue;
}  // namespace internal

// EXPERIMENTAL: API MAY BE SUBJECTED TO SUDDEN CHANGES.
//
// Shared batch scheduler designed for batches which are processed by a serial
// device (e.g. GPU, TPU). When batch processing involves a mix of
// parallelizable cpu work and non-parallelizable on-device work, overall
// latency can be minimized by producing batches at a (load dependent) rate
// which keeps the serial device uniformly busy.
//
// SerialDeviceBatchScheduler (SDBS) controls the batching rate by limiting the
// allowed number of concurrently processed batches. Too large a limit causes
// batches to pile up behind the serial device, adding to the overall batch
// latency. Too small a limit underutilizes the serial device and harms latency
// by forcing batches to wait longer to be processed. Feedback from the device
// (i.e. avg number of batches directly pending on the device) is used to set
// the correct limit.
//
// SDBS groups requests into per model batches which are processed when a batch
// processing thread becomes available. SDBS prioritizes batches primarily by
// age (i.e. the batch's oldest request) along with a configurable preference
// for scheduling larger batches first.


template <typename TaskType>
class SerialDeviceBatchScheduler : public std::enable_shared_from_this<
                                       SerialDeviceBatchScheduler<TaskType>> {
 public:
  ~SerialDeviceBatchScheduler();

  struct Options {
    // The name to use for the pool of batch threads.
    string thread_pool_name = {"batch_threads"};
    // Maximum number of batch processing threads.
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
    int64 initial_in_flight_batches_limit = 3;
    // Returns the current number of batches directly waiting to be processed
    // by the serial device (i.e. GPU, TPU).
    std::function<int64()> get_pending_on_serial_device;
    // Desired average number of batches directly waiting to be processed by the
    // serial device. Small numbers of O(1) should deliver the best latency.
    double target_pending = 2;
    // Number of batches between potential adjustments of
    // in_flight_batches_limit.  Larger numbers will reduce noise, but will be
    // less responsive to sudden changes in workload.
    int64 batches_to_average_over = 1000;
  };

  // Ownership is shared between the caller of Create() and any queues created
  // via AddQueue().
  static Status Create(
      const Options& options,
      std::shared_ptr<SerialDeviceBatchScheduler<TaskType>>* scheduler);

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

  double recent_low_traffic_ratio() {
    mutex_lock l(mu_);
    return recent_low_traffic_ratio_;
  }

 private:
  // access to AddBatch(), RemoveQueue(), env().
  friend class internal::SDBSQueue<TaskType>;

  explicit SerialDeviceBatchScheduler(const Options& options);

  // Continuously retrieves and processes batches.
  void ProcessBatches();

  // Notifies scheduler of non-empty batch which is eligible for processing.
  void AddBatch(const internal::SDBSBatch<TaskType>* batch);

  // Removes queue from scheduler.
  void RemoveQueue(const internal::SDBSQueue<TaskType>* queue);

  Env* env() const { return options_.env; }

  const Options options_;

  // Collection of batches added by AddBatch. Owned by scheduler until they are
  // released for processing.
  std::vector<const internal::SDBSBatch<TaskType>*> batches_ TF_GUARDED_BY(mu_);

  // Unowned queues and callbacks added by AddQueue.
  std::unordered_map<const internal::SDBSQueue<TaskType>*, BatchProcessor>
      queues_and_callbacks_ TF_GUARDED_BY(mu_);

  // Responsible for running the batch processing callbacks.
  std::unique_ptr<thread::ThreadPool> batch_thread_pool_;

  // Limit on number of batches which can be concurrently processed.
  int64 in_flight_batches_limit_ TF_GUARDED_BY(mu_);

  // Number of batch processing threads.
  int64 processing_threads_ TF_GUARDED_BY(mu_) = 0;

  // Number of batches processed since the last in_flight_batches_limit_
  // adjustment.
  int64 batch_count_ TF_GUARDED_BY(mu_) = 0;

  // Number of times since the last in_flight_batches_limit_ adjustment when a
  // processing thread was available but there were no batches to process.
  int64 no_batch_count_ TF_GUARDED_BY(mu_) = 0;

  // Sum of batches pending on the serial device since the last
  // in_flight_batches_limit_ adjustment.
  int64 pending_sum_ = 0;

  // Sum of batch latencies since the last in_flight_batches_limit_ adjustment.
  int64 batch_latency_sum_ = 0;

  // Average period between which two consecutive batches begin processing.
  int64 batch_period_micros_ = 0;

  // Moving average tracking the fraction of recent in_flight_batches_limit_
  // adjustments where the external traffic was not high enough to provide
  // useful feedback for an adjustment.
  double recent_low_traffic_ratio_ = 0;

  mutex mu_;

  TF_DISALLOW_COPY_AND_ASSIGN(SerialDeviceBatchScheduler);
};

//////////////////////////////////////////////////////////
// Implementation details follow. API users need not read.

namespace internal {
// Consolidates tasks into batches, passing them off to the
// SerialDeviceBatchScheduler for processing.
template <typename TaskType>
class SDBSQueue : public BatchScheduler<TaskType> {
 public:
  using QueueOptions =
      typename SerialDeviceBatchScheduler<TaskType>::QueueOptions;

  SDBSQueue(std::shared_ptr<SerialDeviceBatchScheduler<TaskType>> scheduler,
            const QueueOptions& options);

  ~SDBSQueue() override;

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
  void ReleaseBatch(const SDBSBatch<TaskType>* batch);

  size_t max_task_size() const override { return options_.max_batch_size; }

 private:
  std::shared_ptr<SerialDeviceBatchScheduler<TaskType>> scheduler_;
  const QueueOptions options_;
  // Owned by scheduler_.
  SDBSBatch<TaskType>* current_batch_ TF_GUARDED_BY(mu_) = nullptr;
  int64 num_enqueued_batches_ TF_GUARDED_BY(mu_) = 0;
  int64 num_enqueued_tasks_ TF_GUARDED_BY(mu_) = 0;
  mutable mutex mu_;
  TF_DISALLOW_COPY_AND_ASSIGN(SDBSQueue);
};

// Batch which remembers when and by whom it was created.
template <typename TaskType>
class SDBSBatch : public Batch<TaskType> {
 public:
  SDBSBatch(SDBSQueue<TaskType>* queue, int64_t creation_time_micros)
      : queue_(queue), creation_time_micros_(creation_time_micros) {}

  ~SDBSBatch() override {}

  SDBSQueue<TaskType>* queue() const { return queue_; }

  int64 creation_time_micros() const { return creation_time_micros_; }

 private:
  SDBSQueue<TaskType>* queue_;
  const int64 creation_time_micros_;
  TF_DISALLOW_COPY_AND_ASSIGN(SDBSBatch);
};
}  // namespace internal

// ---------------- SerialDeviceBatchScheduler ----------------

template <typename TaskType>
Status SerialDeviceBatchScheduler<TaskType>::Create(
    const Options& options,
    std::shared_ptr<SerialDeviceBatchScheduler<TaskType>>* scheduler) {
  if (options.num_batch_threads < 1) {
    return errors::InvalidArgument("num_batch_threads must be positive; was ",
                                   options.num_batch_threads);
  }
  if (options.initial_in_flight_batches_limit < 1) {
    return errors::InvalidArgument(
        "initial_in_flight_batches_limit must be positive; was ",
        options.initial_in_flight_batches_limit);
  }
  if (options.initial_in_flight_batches_limit > options.num_batch_threads) {
    return errors::InvalidArgument(
        "initial_in_flight_batches_limit (",
        options.initial_in_flight_batches_limit,
        ") should not be larger than num_batch_threads (",
        options.num_batch_threads, ")");
  }
  if (options.full_batch_scheduling_boost_micros < 0) {
    return errors::InvalidArgument(
        "full_batch_scheduling_boost_micros can't be negative; was ",
        options.full_batch_scheduling_boost_micros);
  }
  if (options.batches_to_average_over < 1) {
    return errors::InvalidArgument(
        "batches_to_average_over should be "
        "greater than or equal to 1; was ",
        options.batches_to_average_over);
  }
  if (options.target_pending <= 0) {
    return errors::InvalidArgument(
        "target_pending should be larger than zero; was ",
        options.target_pending);
  }
  if (!options.get_pending_on_serial_device) {
    return errors::InvalidArgument(
        "get_pending_on_serial_device must be "
        "specified");
  }
  scheduler->reset(new SerialDeviceBatchScheduler<TaskType>(options));
  return Status::OK();
}

template <typename TaskType>
SerialDeviceBatchScheduler<TaskType>::SerialDeviceBatchScheduler(
    const Options& options)
    : options_(options),
      in_flight_batches_limit_(options.initial_in_flight_batches_limit),
      processing_threads_(options.initial_in_flight_batches_limit) {
  batch_thread_pool_.reset(new thread::ThreadPool(
      env(), options.thread_pool_name, options.num_batch_threads));
  for (int i = 0; i < processing_threads_; i++) {
    batch_thread_pool_->Schedule(
        std::bind(&SerialDeviceBatchScheduler<TaskType>::ProcessBatches, this));
  }
}

template <typename TaskType>
SerialDeviceBatchScheduler<TaskType>::~SerialDeviceBatchScheduler() {
  // Signal processing threads to exit.
  {
    mutex_lock l(mu_);
    processing_threads_ = 0;
  }
  // Hangs until all threads finish.
  batch_thread_pool_.reset();
}

template <typename TaskType>
Status SerialDeviceBatchScheduler<TaskType>::AddQueue(
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
  internal::SDBSQueue<TaskType>* SDBS_queue_raw;
  queue->reset(SDBS_queue_raw = new internal::SDBSQueue<TaskType>(
                   this->shared_from_this(), options));
  mutex_lock l(mu_);
  queues_and_callbacks_[SDBS_queue_raw] = process_batch_callback;
  return Status::OK();
}

template <typename TaskType>
void SerialDeviceBatchScheduler<TaskType>::AddBatch(
    const internal::SDBSBatch<TaskType>* batch) {
  mutex_lock l(mu_);
  batches_.push_back(batch);
}

template <typename TaskType>
void SerialDeviceBatchScheduler<TaskType>::RemoveQueue(
    const internal::SDBSQueue<TaskType>* queue) {
  mutex_lock l(mu_);
  queues_and_callbacks_.erase(queue);
}

template <typename TaskType>
void SerialDeviceBatchScheduler<TaskType>::ProcessBatches() {
  const int64 kIdleThreadSleepTimeMicros = 1000;
  const double kMaxNoBatchRatio = .1;
  const double kLowTrafficMovingAverageFactor = .1;
  for (;;) {
    mu_.lock();
    if (processing_threads_ < 1 ||
        processing_threads_ > in_flight_batches_limit_) {
      processing_threads_--;
      mu_.unlock();
      break;
    }
    if (batches_.empty()) {
      no_batch_count_++;
      int64_t sleep_time = batch_period_micros_ ? batch_period_micros_
                                                : kIdleThreadSleepTimeMicros;
      mu_.unlock();
      env()->SleepForMicroseconds(sleep_time);
      continue;
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
    const internal::SDBSBatch<TaskType>* batch = *best_it;
    batches_.erase(best_it);
    // Queue may destroy itself after ReleaseBatch is called.
    batch->queue()->ReleaseBatch(batch);
    auto callback = queues_and_callbacks_[batch->queue()];
    mu_.unlock();
    int64_t start_time = env()->NowMicros();
    callback(std::unique_ptr<Batch<TaskType>>(
        const_cast<internal::SDBSBatch<TaskType>*>(batch)));
    int64_t end_time = env()->NowMicros();
    mu_.lock();
    batch_count_++;
    batch_latency_sum_ += end_time - start_time;
    pending_sum_ += options_.get_pending_on_serial_device();
    if (batch_count_ == options_.batches_to_average_over) {
      recent_low_traffic_ratio_ *= (1 - kLowTrafficMovingAverageFactor);
      // Only adjust in_flight_batches_limit_ if external load is large enough
      // to consistently provide batches. Otherwise we would (mistakenly) assume
      // that the device is underutilized because in_flight_batches_limit_ is
      // too small.
      if (no_batch_count_ < kMaxNoBatchRatio * batch_count_) {
        double avg_pending = pending_sum_ / static_cast<double>(batch_count_);
        // Avg processing time / # of concurrent batches gives the avg period
        // between which two consecutive batches begin processing. Used to set a
        // reasonable sleep time for idle batch processing threads.
        batch_period_micros_ =
            batch_latency_sum_ / batch_count_ / in_flight_batches_limit_;
        // When the processing pipeline is consistently busy, the average number
        // of pending batches differs from in_flight_batches_limit_ by a
        // load-dependent offset. Adjust in_flight_batches_limit_to maintain
        // the desired target pending.
        in_flight_batches_limit_ +=
            std::round(options_.target_pending - avg_pending);
        in_flight_batches_limit_ = std::max(in_flight_batches_limit_, int64{1});
        in_flight_batches_limit_ =
            std::min(in_flight_batches_limit_, options_.num_batch_threads);
        // Add extra processing threads if necessary.
        if (processing_threads_ > 0 &&
            processing_threads_ < in_flight_batches_limit_) {
          int extra_threads = in_flight_batches_limit_ - processing_threads_;
          for (int i = 0; i < extra_threads; i++) {
            batch_thread_pool_->Schedule(std::bind(
                &SerialDeviceBatchScheduler<TaskType>::ProcessBatches, this));
          }
          processing_threads_ = in_flight_batches_limit_;
        }
      } else {
        recent_low_traffic_ratio_ += kLowTrafficMovingAverageFactor;
      }
      batch_count_ = 0;
      no_batch_count_ = 0;
      pending_sum_ = 0;
      batch_latency_sum_ = 0;
    }
    mu_.unlock();
  }
}

// ---------------- SDBSQueue ----------------

namespace internal {
template <typename TaskType>
SDBSQueue<TaskType>::SDBSQueue(
    std::shared_ptr<SerialDeviceBatchScheduler<TaskType>> scheduler,
    const QueueOptions& options)
    : scheduler_(scheduler), options_(options) {}

template <typename TaskType>
SDBSQueue<TaskType>::~SDBSQueue() {
  // Wait until last batch has been scheduled.
  const int kSleepMicros = 1000;
  for (;;) {
    {
      mutex_lock l(mu_);
      if (num_enqueued_batches_ == 0) {
        break;
      }
    }
    scheduler_->env()->SleepForMicroseconds(kSleepMicros);
  }
  scheduler_->RemoveQueue(this);
}

template <typename TaskType>
Status SDBSQueue<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  SDBSBatch<TaskType>* new_batch = nullptr;
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
          new SDBSBatch<TaskType>(this, scheduler_->env()->NowMicros());
    }
    current_batch_->AddTask(std::move(*task));
    num_enqueued_tasks_++;
  }
  // AddBatch must be called outside of lock, since it may call ReleaseBatch.
  if (new_batch != nullptr) scheduler_->AddBatch(new_batch);
  return Status::OK();
}

template <typename TaskType>
void SDBSQueue<TaskType>::ReleaseBatch(const SDBSBatch<TaskType>* batch) {
  mutex_lock l(mu_);
  num_enqueued_batches_--;
  num_enqueued_tasks_ -= batch->num_tasks();
  if (batch == current_batch_) {
    current_batch_->Close();
    current_batch_ = nullptr;
  }
}

template <typename TaskType>
size_t SDBSQueue<TaskType>::NumEnqueuedTasks() const {
  mutex_lock l(mu_);
  return num_enqueued_tasks_;
}

template <typename TaskType>
size_t SDBSQueue<TaskType>::SchedulingCapacity() const {
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

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_SERIAL_DEVICE_BATCH_SCHEDULER_H_
