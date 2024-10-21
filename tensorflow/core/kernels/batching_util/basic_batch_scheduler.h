/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BASIC_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BASIC_BATCH_SCHEDULER_H_

#include <stddef.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"

namespace tensorflow {
namespace serving {

// A BatchScheduler implementation geared toward handling a single request type
// running on a specific set of hardware resources. A typical scenario is one in
// which all requests invoke the same machine-learned model on one GPU.
//
// If there are, say, two GPUs and two models each bound to one of the GPUs, one
// could use two BasicBatchScheduler instances to schedule the two model/GPU
// combinations independently. If multiple models must share a given GPU or
// other hardware resource, consider using SharedBatchScheduler instead.
//
//
// PARAMETERS AND BEHAVIOR:
//
// BasicBatchScheduler runs a fixed pool of threads, which it uses to process
// batches of tasks. It enforces a maximum batch size, and enqueues a bounded
// number of tasks. If the queue is nearly empty, such that a full batch cannot
// be formed, when a thread becomes free, it anyway schedules a batch
// immediately if a task has been in the queue for longer than a given timeout
// parameter. If the timeout parameter is set to 0, then the batch threads will
// always be kept busy (unless there are zero tasks waiting to be processed).
//
// For online serving, it is recommended to set the maximum number of enqueued
// batches worth of tasks equal to the number of batch threads, which allows
// enqueuing of enough tasks s.t. if every thread becomes available it can be
// kept busy, but no more. For bulk processing jobs and throughput-oriented
// benchmarks, you may want to set it much higher.
//
// When Schedule() is called, if the queue is full the call will fail with an
// UNAVAILABLE error (after which the client may retry again later). If the call
// succeeds, the maximum time the task will spend in the queue before being
// placed in a batch and assigned to a thread for processing, is the greater of:
//  - the maximum time to process ceil(max_enqueued_batches/num_batch_threads)
//    (1 in the recommended configuration) batches of previously-submitted tasks
//  - the configured timeout parameter (which can be 0, as mentioned above)
//
// Unlike StreamingBatchScheduler, when BasicBatchScheduler assigns a batch to a
// thread, it closes the batch. The process-batch callback may assume that every
// batch it receives is closed at the outset.
//
//
// RECOMMENDED USE-CASES:
//
// BasicBatchScheduler is suitable for use-cases that feature a single kind of
// request (e.g. a server performing inference with a single machine-learned
// model, possibly evolving over time), with loose versioning semantics.
// Concretely, the following conditions should hold:
//
//  A. All requests batched onto a given resource (e.g. a hardware accelerator,
//     or a pool accelerators) are of the same type. For example, they all
//     invoke the same machine-learned model.
//
//     These variations are permitted:
//      - The model may reside in a single servable, or it may be spread across
//        multiple servables that are used in unison (e.g. a vocabulary lookup
//        table servable and a tensorflow session servable).
//      - The model's servable(s) may be static, or they may evolve over time
//        (successive servable versions).
//      - Zero or more of the servables are used in the request thread; the rest
//        are used in the batch thread. In our running example, the vocabulary
//        lookups and tensorflow runs may both be performed in the batch thread,
//        or alternatively the vocabulary lookup may occur in the request thread
//        with only the tensorflow run performed in the batch thread.
//
//     In contrast, BasicBatchScheduler is not a good fit if the server
//     hosts multiple distinct models running on a pool accelerators, with each
//     request specifying which model it wants to use. BasicBatchScheduler
//     has no facility to time-multiplex the batch threads across multiple
//     models in a principled way. More basically, it cannot ensure that a given
//     batch doesn't contain a mixture of requests for different models.
//
//  B. Requests do not specify a particular version of the servable(s) that must
//     be used. Instead, each request is content to use the "latest" version.
//
//     BasicBatchScheduler does not constrain which requests get grouped
//     together into a batch, so using this scheduler there is no way to achieve
//     cohesion of versioned requests to version-specific batches.
//
//  C. No servable version coordination needs to be performed between the
//     request threads and the batch threads. Often, servables are only used in
//     the batch threads, in which case this condition trivially holds. If
//     servables are used in both threads, then the use-case must tolerate
//     version skew across the servables used in the two kinds of threads.
//
//
// EXAMPLE USE-CASE FLOW:
//
// For such use-cases, request processing via BasicBatchScheduler generally
// follows this flow (given for illustration; variations are possible):
//  1. Optionally perform some pre-processing on each request in the request
//     threads.
//  2. Route the requests to the batch scheduler, as batching::Task objects.
//     (Since all requests are of the same type and are not versioned, the
//     scheduler is free to group them into batches arbitrarily.)
//  3. Merge the requests into a single batched representation B.
//  4. Obtain handles to the servable(s) needed to process B. The simplest
//     approach is to obtain the latest version of each servable. Alternatively,
//     if cross-servable consistency is required (e.g. the vocabulary lookup
//     table's version number must match that of the tensorflow session),
//     identify an appropriate version number and obtain the servable handles
//     accordingly.
//  5. Process B using the obtained servable handles, and split the result into
//     individual per-request units.
//  6. Perform any post-processing in the batch thread and/or request thread.
//
//
// PERFORMANCE TUNING: See README.md.
//
template <typename TaskType>
class BasicBatchScheduler : public BatchScheduler<TaskType> {
 public:
  // TODO(b/25089730): Tune defaults based on best practices as they develop.
  // (Keep them mirrored to the ones in SharedBatchScheduler::QueueOptions and
  // SharedBatchScheduler::Options.)
  struct Options {
    // Options related with (underlying) shared batch scheduler.
    // 'thread_pool_name' and 'num_batch_threads' are used to initialize
    // a shared batch scheduler underlyingly iff 'shared_batch_scheduler' is
    // nullptr.
    //
    // There are two ways to specify threading:
    // 1) Have each session create its own pool.
    // 2) Have multiple sessions share the same pool.
    //
    // In general, the number of threads should be tied to roughly the number of
    // compute resources (CPU cores or accelerator cores) backing the threads.
    // Sharing a thread pool helps alleviate potential over allocation of
    // threads to limited compute resources.

    // To have each session create its own thread pool (1) set
    // thread_pool_name/num_batch_threads.

    // To share a thread pool (2) create a scheduler and pass it in.

    // The name to use for the pool of batch threads.
    string thread_pool_name = {"batch_threads"};

    // The number of threads to use to process batches.
    // Must be >= 1, and should be tuned carefully.
    int num_batch_threads = port::MaxParallelism();

    // If specified, this scheduler will be used underlyingly to schedule
    // batches. Note setting this means `thread_pool_name` and
    // `num_batch_threads` are ignored.
    std::shared_ptr<SharedBatchScheduler<TaskType>> shared_batch_scheduler =
        nullptr;

    // Options for queue.
    // The maximum size of each batch.
    //
    // The scheduler may form batches of any size between 1 and this number
    // (inclusive). If there is a need to quantize the batch sizes, i.e. only
    // submit batches whose size is in a small set of allowed sizes, that can be
    // done by adding padding in the process-batch callback.
    int max_batch_size = 1000;

    // If a task has been enqueued for this amount of time (in microseconds),
    // and a thread is available, the scheduler will immediately form a batch
    // from enqueued tasks and assign the batch to the thread for processing,
    // even if the batch's size is below 'max_batch_size'.
    //
    // This parameter offers a way to bound queue latency, so that a task isn't
    // stuck in the queue indefinitely waiting for enough tasks to arrive to
    // make a full batch. (The latency bound is given in the class documentation
    // above.)
    //
    // The goal is to smooth out batch sizes under low request rates, and thus
    // avoid latency spikes.
    int64_t batch_timeout_micros = 0;

    // The maximum allowable number of enqueued (accepted by Schedule() but
    // not yet being processed on a batch thread) tasks in terms of batches.
    // If this limit is reached, Schedule() will return an UNAVAILABLE error.
    // See the class documentation above for guidelines on how to tune this
    // parameter.
    int max_enqueued_batches = 10;

    // If true, an input task (i.e., input of `BasicBatchScheduler::Schedule`)
    // with a large size (i.e., larger than the largest value of
    // `allowed_batch_sizes`) will be split into multiple smaller batch tasks
    // and possibly put into different batches for processing. If false, each
    // input task is put into one batch as a whole for processing.
    //
    // API note:
    // The value of this option doesn't affect processing output given the same
    // input; it affects implementation details as stated below:
    // 1. Improve batching efficiency by eliminating unnecessary padding in the
    // following scenario: when an open batch has M slots while an input of size
    // N is scheduled (M < N), the input can be split to fill remaining slots
    // of an open batch as opposed to padding.
    // 2.`max_batch_size` specifies the limit of input and
    // `max_execution_batch_size` specifies the limit of a task to be processed.
    // API user can give an input of size 128 when 'max_execution_batch_size'
    // is 32 -> implementation can split input of 128 into 4 x 32, schedule
    // concurrent processing, and then return concatenated results corresponding
    // to 128.
    bool enable_large_batch_splitting = false;

    // `split_input_task_func` specifies how to split `input_task` into
    // `output_tasks`.
    //
    // `input_task`: a unit of task to be split.
    // `first_output_task_size`: task size of first output.
    // `max_batch_size`: Maximum size of each batch.
    // `output_tasks`: A list of output tasks after split.
    //
    // REQUIRED:
    // 1) All `output_tasks` should be non-empty tasks.
    // 2) Sizes of `output_tasks` add up to size of `input_task`.
    //
    // NOTE:
    // Instantiations of `TaskType` may vary, so it's up to caller to define
    // how (e.g., which members to access) to split input tasks.
    std::function<absl::Status(
        std::unique_ptr<TaskType>* input_task, int first_output_task_size,
        int input_batch_size_limit,
        std::vector<std::unique_ptr<TaskType>>* output_tasks)>
        split_input_task_func;

    // The maximum size of each enqueued batch (i.e., in `batches_`).
    //
    // The scheduler may form batches of any size between 1 and this number
    // (inclusive). If there is a need to quantize the batch sizes, i.e. only
    // submit batches whose size is in a small set of allowed sizes, that can be
    // done by adding padding in the process-batch callback.
    //
    // REQUIRES:
    // - If enable_large_batch_splitting is true, `max_execution_batch_size` is
    // less than or equal to `max_batch_size`.
    // - If enable_large_batch_splitting is false, `max_execution_batch_size` is
    // equal to `max_batch_size`.
    int max_execution_batch_size = 10;

    // The following options are typically only overridden by test code.

    // The environment to use.
    Env* env = Env::Default();
  };
  static absl::Status Create(
      const Options& options,
      std::function<void(std::unique_ptr<Batch<TaskType>>)>
          process_batch_callback,
      std::unique_ptr<BasicBatchScheduler>* scheduler);

  ~BasicBatchScheduler() override = default;

  absl::Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;

  size_t max_task_size() const override {
    return shared_scheduler_queue_->max_task_size();
  }

 private:
  explicit BasicBatchScheduler(
      std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue);

  // This class is merely a thin wrapper around a SharedBatchScheduler with a
  // single queue.
  std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue_;

  BasicBatchScheduler(const BasicBatchScheduler&) = delete;
  void operator=(const BasicBatchScheduler&) = delete;
};

//////////
// Implementation details follow. API users need not read.

template <typename TaskType>
absl::Status BasicBatchScheduler<TaskType>::Create(
    const Options& options,
    std::function<void(std::unique_ptr<Batch<TaskType>>)>
        process_batch_callback,
    std::unique_ptr<BasicBatchScheduler>* scheduler) {
  std::shared_ptr<SharedBatchScheduler<TaskType>> shared_scheduler;

  if (options.shared_batch_scheduler == nullptr) {
    typename SharedBatchScheduler<TaskType>::Options shared_scheduler_options;
    shared_scheduler_options.thread_pool_name = options.thread_pool_name;
    shared_scheduler_options.num_batch_threads = options.num_batch_threads;
    shared_scheduler_options.env = options.env;

    TF_RETURN_IF_ERROR(SharedBatchScheduler<TaskType>::Create(
        shared_scheduler_options, &shared_scheduler));
  } else {
    shared_scheduler = options.shared_batch_scheduler;
  }

  typename SharedBatchScheduler<TaskType>::QueueOptions
      shared_scheduler_queue_options;
  shared_scheduler_queue_options.input_batch_size_limit =
      options.max_batch_size;
  shared_scheduler_queue_options.batch_timeout_micros =
      options.batch_timeout_micros;
  shared_scheduler_queue_options.max_enqueued_batches =
      options.max_enqueued_batches;
  shared_scheduler_queue_options.enable_large_batch_splitting =
      options.enable_large_batch_splitting;
  shared_scheduler_queue_options.split_input_task_func =
      options.split_input_task_func;
  shared_scheduler_queue_options.max_execution_batch_size =
      options.max_execution_batch_size;
  std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue;
  TF_RETURN_IF_ERROR(shared_scheduler->AddQueue(shared_scheduler_queue_options,
                                                process_batch_callback,
                                                &shared_scheduler_queue));

  scheduler->reset(
      new BasicBatchScheduler<TaskType>(std::move(shared_scheduler_queue)));
  return absl::OkStatus();
}

template <typename TaskType>
absl::Status BasicBatchScheduler<TaskType>::Schedule(
    std::unique_ptr<TaskType>* task) {
  return shared_scheduler_queue_->Schedule(task);
}

template <typename TaskType>
size_t BasicBatchScheduler<TaskType>::NumEnqueuedTasks() const {
  return shared_scheduler_queue_->NumEnqueuedTasks();
}

template <typename TaskType>
size_t BasicBatchScheduler<TaskType>::SchedulingCapacity() const {
  return shared_scheduler_queue_->SchedulingCapacity();
}

template <typename TaskType>
BasicBatchScheduler<TaskType>::BasicBatchScheduler(
    std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue)
    : shared_scheduler_queue_(std::move(shared_scheduler_queue)) {}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BASIC_BATCH_SCHEDULER_H_
