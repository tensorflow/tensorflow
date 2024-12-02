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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_SHARED_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_SHARED_BATCH_SCHEDULER_H_

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "tensorflow/core/kernels/batching_util/batch_input_task.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler_utils.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"
#include "tensorflow/core/kernels/batching_util/periodic_function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tsl/platform/criticality.h"
#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace serving {
namespace internal {
template <typename TaskType>
class Queue;
}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

namespace tensorflow {
namespace serving {

// A batch scheduler for server instances that service multiple request types
// (e.g. multiple machine-learned models, or multiple versions of a model served
// concurrently), or even multiple distinct tasks for a given request. The
// scheduler multiplexes batches of different kinds of tasks onto a fixed-size
// thread pool (each batch contains tasks of a single type), in a carefully
// controlled manner. A common configuration is to set the number of threads
// equal to the number of hardware accelerator units, in which case the
// scheduler takes care of multiplexing the task types onto the shared hardware,
// in a manner that is both fair and efficient.
//
// Semantically, SharedBatchScheduler behaves like having N instances of
// BasicBatchScheduler (see basic_batch_scheduler.h), one per task type. The
// difference is that under the covers there is a single shared thread pool,
// instead of N independent ones, with their sharing deliberately coordinated.
//
// SharedBatchScheduler does not implement the BatchScheduler API; rather, it
// presents an abstraction of "queues", where each queue corresponds to one type
// of task. Tasks submitted to a given queue are placed in their own batches,
// and cannot be mixed with other tasks. Queues can be added and deleted
// dynamically, to accommodate e.g. versions of a model being brought up and
// down over the lifetime of a server.
//
// The batch thread pool round-robins through the queues, running one batch
// from a queue and then moving to the next queue. Each queue behaves like a
// BasicBatchScheduler instance, in the sense that it has maximum batch size and
// timeout parameters, which govern when a batch is eligible to be processed.
//
// Each queue is independently configured with a maximum size (in terms of the
// maximum number of batches worth of enqueued tasks). For online serving, it is
// recommended that the queue sizes be configured such that the sum of the sizes
// of the active queues roughly equal the number of batch threads. (The idea is
// that if all threads become available at roughly the same time, there will be
// enough enqueued work for them to take on, but no more.)
//
// If queue sizes are configured in the manner suggested above, the maximum time
// a task can spend in a queue before being placed in a batch and assigned to a
// thread for processing, is the greater of:
//  - the maximum time to process one batch of tasks from any active queue
//  - the configured timeout parameter for the task's queue (which can be 0)
//
// For bulk processing jobs and throughput-oriented benchmarks, you may want to
// set the maximum queue size to a large value.
//
// TODO(b/26539183): Support queue servicing policies other than round-robin.
// E.g. let each queue specify a "share" (an int >= 1), so e.g. with queues A
// and B having shares 1 and 2 respectively, the servicing pattern is ABBABB...
//
//
// PERFORMANCE TUNING: See README.md.
//
template <typename TaskType>
class SharedBatchScheduler
    : public std::enable_shared_from_this<SharedBatchScheduler<TaskType>> {
 public:
  using BatchTaskUniquePtr = std::unique_ptr<Batch<TaskType>>;

  using ProcessBatchCallback =
      std::variant<std::function<void(BatchTaskUniquePtr)>,
                   std::function<void(BatchTaskUniquePtr,
                                      std::vector<std::unique_ptr<TaskType>>)>>;
  // TODO(b/25089730): Tune defaults based on best practices as they develop.
  struct Options {
    // The name to use for the pool of batch threads.
    string thread_pool_name = {"batch_threads"};

    // The number of threads to use to process batches.
    // Must be >= 1, and should be tuned carefully.
    int num_batch_threads = port::MaxParallelism();

    // The environment to use.
    // (Typically only overridden by test code.)
    Env* env = Env::Default();

    // If true, when multiple queues have available batches to process, they
    // will be prioritized based on a (priority, arrival_time) key.
    bool rank_queues = false;

    // If true, Create() will return a global instance of the scheduler. Only
    // the options provided in the first Create() call will be used to
    // initialize the global scheduler.
    bool use_global_scheduler = false;
  };
  // Ownership is shared between the caller of Create() and any queues created
  // via AddQueue().
  static absl::Status Create(
      const Options& options,
      std::shared_ptr<SharedBatchScheduler<TaskType>>* scheduler);

  virtual ~SharedBatchScheduler();

  // Adds a queue to which tasks may be submitted. The returned queue implements
  // the BatchScheduler API. Each queue has its own set of scheduling options,
  // and its own callback to process batches of tasks submitted to the queue.
  //
  // The returned queue's destructor blocks until all tasks submitted to it have
  // been processed.
  struct QueueOptions {
    // The size limit of an input batch to the queue.
    //
    // If `enable_large_batch_splitting` is True, 'input_batch_size_limit'
    // should be greater or equal than `max_execution_batch_size`; otherwise
    // `input_batch_size_limit` should be equal to `max_execution_batch_size`.
    size_t input_batch_size_limit = 1000;

    // If a task has been enqueued for this amount of time (in microseconds),
    // and a thread is available, the scheduler will immediately form a batch
    // from enqueued tasks and assign the batch to the thread for processing,
    // even if the batch's size is below 'input_batch_size_limit'.
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
    //
    // Must be positive, or else invalid argument error will be returned at
    // queue creation time.
    size_t max_enqueued_batches = 10;

    // If true, queue implementation would split one input batch task into
    // subtasks (as specified by `split_input_task_func` below) and fit subtasks
    // into different batches.
    //
    // For usage of `split_input_task_func`, please see its comment.
    bool enable_large_batch_splitting = false;

    // `input_task`: a unit of task to be split.
    // `first_output_task_size`: task size of first output.
    // `max_execution_batch_size`: Maximum size of each batch.
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

    // The maximum size of each enqueued batch (i.e., in
    // `high_priority_batches_`).
    //
    // The scheduler may form batches of any size between 1 and this number
    // (inclusive). If there is a need to quantize the batch sizes, i.e. only
    // submit batches whose size is in a small set of allowed sizes, that can be
    // done by adding padding in the process-batch callback.
    size_t max_execution_batch_size = 1000;

    // If non-empty, contains configured batch sizes.
    std::vector<int32> allowed_batch_sizes;

    // If true, the padding will not be appended.
    bool disable_padding = false;

    // The padding policy to use.
    //
    // See the documentation for kPadUpPolicy for details.
    string batch_padding_policy = string(kPadUpPolicy);

    // A pointer to a ModelBatchStats instance for this model. To be used for
    // cost-based padding policy selection.
    //
    // If null, some other padding policy will be used if a cost-based one is
    // requested.
    ModelBatchStats* model_batch_stats = nullptr;

    // If true, queue implementation would split high priority and low priority
    // inputs into two sub queues.
    bool enable_priority_queue = false;

    // A separate set of queue options for different priority inputs.
    // Use iff `enable_priority_queue` is true.
    struct PriorityQueueOptions {
      // See QueueOptions.max_execution_batch_size
      size_t max_execution_batch_size = 0;
      // See QueueOptions.batch_timeout_micros
      int64_t batch_timeout_micros = 0;
      // See QueueOptions.input_batch_size_limit
      size_t input_batch_size_limit = 0;
      // See QueueOptions.max_enqueued_batches
      size_t max_enqueued_batches = 0;
      // See QueueOptions.allowed_batch_sizes
      std::vector<int32> allowed_batch_sizes;
    };
    // A subset of queue options for high priority input. These options are
    // currently not being used in favor of the equivalents options at the
    // QueueOptions level.
    PriorityQueueOptions high_priority_queue_options;
    // A subset of queue options for low priority input.
    PriorityQueueOptions low_priority_queue_options;

    // A policy that determines the mixed priority batching behavior. It is
    // effective only when enable_priority_queue is true.
    MixedPriorityBatchingPolicy mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;
  };
  // This method is marked virtual for testing purposes only.
  virtual absl::Status AddQueue(
      const QueueOptions& options, ProcessBatchCallback process_batch_callback,
      std::unique_ptr<BatchScheduler<TaskType>>* queue);

 protected:
  explicit SharedBatchScheduler(const Options& options);

 private:
  void GetNextWorkItem_Locked(internal::Queue<TaskType>** queue_for_batch_out,
                              BatchTaskUniquePtr* batch_to_process_out)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // The code executed in 'batch_threads_'. Obtains a batch to process from the
  // queue pointed to by 'next_queue_to_schedule_', and processes it. If that
  // queue declines to provide a batch to process, moves onto the next queue. If
  // no queues provide a batch to process, just sleeps briefly and exits.
  void ThreadLogic();

  // Called by `AddQueue`.
  absl::Status AddQueueAfterRewritingOptions(
      const QueueOptions& options, ProcessBatchCallback process_batch_callback,
      std::unique_ptr<BatchScheduler<TaskType>>* queue);

  static bool BatchExists(const BatchTaskUniquePtr& batch_to_process);

  const Options options_;

  mutex mu_;

  // A list of queues. (We use std::list instead of std::vector to ensure that
  // iterators are not invalidated by adding/removing elements. It also offers
  // efficient removal of elements from the middle.)
  using QueueList = std::list<std::unique_ptr<internal::Queue<TaskType>>>;

  // All "active" queues, i.e. ones that either:
  //  - have not been removed, or
  //  - have been removed but are not yet empty.
  QueueList queues_ TF_GUARDED_BY(mu_);

  // An iterator over 'queues_', pointing to the queue from which the next
  // available batch thread should grab work.
  typename QueueList::iterator next_queue_to_schedule_ TF_GUARDED_BY(mu_);

  // Used by idle batch threads to wait for work to enter the system. Notified
  // whenever a batch becomes schedulable.
  condition_variable schedulable_batch_cv_;

  // Threads that process batches obtained from the queues.
  std::vector<std::unique_ptr<PeriodicFunction>> batch_threads_;

  SharedBatchScheduler(const SharedBatchScheduler&) = delete;
  void operator=(const SharedBatchScheduler&) = delete;
};

//////////
// Implementation details follow. API users need not read.

namespace internal {

// A task queue for SharedBatchScheduler. Accepts tasks and accumulates them
// into batches, and dispenses those batches to be processed via a "pull"
// interface. The queue's behavior is governed by maximum batch size, timeout
// and maximum queue length parameters; see their documentation in
// SharedBatchScheduler.
//
// The queue is implemented as a deque of batches, with these invariants:
//  - The number of batches is between 1 and 'options_.max_enqueued_batches'.
//  - The back-most batch is open; the rest are closed.
//
// Submitted tasks are added to the open batch. If that batch doesn't have room
// but the queue isn't full, then that batch is closed and a new open batch is
// started.
//
// Batch pull requests are handled by dequeuing the front-most batch if it is
// closed. If the front-most batch is open (i.e. the queue contains only one
// batch) and has reached the timeout, it is immediately closed and returned;
// otherwise no batch is returned for the request.
template <typename TaskType>
class Queue {
 public:
  using ProcessBatchCallbackWithoutPaddingTasks =
      std::function<void(std::unique_ptr<Batch<TaskType>>)>;
  using ProcessBatchCallbackWithPaddingTasks =
      std::function<void(std::unique_ptr<Batch<TaskType>>,
                         std::vector<std::unique_ptr<TaskType>>)>;
  using ProcessBatchCallback =
      std::variant<ProcessBatchCallbackWithoutPaddingTasks,
                   ProcessBatchCallbackWithPaddingTasks>;

  using SchedulableBatchCallback = std::function<void()>;
  using SplitInputTaskIntoSubtasksCallback = std::function<absl::Status(
      std::unique_ptr<TaskType>* input_task, int open_batch_remaining_slot,
      int max_execution_batch_size,
      std::vector<std::unique_ptr<TaskType>>* output_tasks)>;
  // Orderable key representing the priority of a batch. Higher priority
  // batches will be prioritized for execution first (when using
  // rank_queues=true).
  // - A smaller key value is higher priority than a larger one.
  // - This is a pair formed from <priority, batch_timestamp>. The exact values
  //   used are an implementation detail of PeekBatchPriority().
  using BatchPriorityKey = std::pair<int, int64_t>;

  Queue(const typename SharedBatchScheduler<TaskType>::QueueOptions& options,
        Env* env, ProcessBatchCallback process_batch_callback,
        SchedulableBatchCallback schedulable_batch_callback);

  // Illegal to destruct unless the queue is empty.
  ~Queue();

  // Submits a task to the queue, with the same semantics as
  // BatchScheduler::Schedule().
  absl::Status Schedule(std::unique_ptr<TaskType>* task);

  // Returns the number of enqueued tasks, with the same semantics as
  // BatchScheduler::NumEnqueuedTasks().
  size_t NumEnqueuedTasks() const;

  // Returns the queue capacity, with the same semantics as
  // BatchScheduler::SchedulingCapacity().
  size_t SchedulingCapacity() const;

  // Returns the maximum allowed size of tasks submitted to the queue.
  size_t max_task_size() const { return options_.input_batch_size_limit; }

  // Returns the maximum allowed size of tasks to be executed.
  // Returned value would be less than or equal to the maximum allowed input
  // size that's provided by caller of batch scheduler.
  size_t max_execution_batch_size() const { return max_execution_batch_size_; }

  // Called by a thread that is ready to process a batch, to request one from
  // this queue. Either returns a batch that is ready to be processed, or
  // nullptr if the queue declines to schedule a batch at this time. If it
  // returns a batch, the batch is guaranteed to be closed.
  typename SharedBatchScheduler<TaskType>::BatchTaskUniquePtr ScheduleBatch();

  // Without mutating the queue, checks if ScheduleBatch() will return a valid
  // batch and if so will return the priority of that batch.
  std::optional<BatchPriorityKey> PeekBatchPriority() const;

  // Retrieves the low priority tasks that can be padded to a high priority
  // batch of the specified size.
  std::vector<std::unique_ptr<TaskType>> GetLowPriorityTasksForPadding(
      size_t batch_size);

  // Processes a batch that has been returned earlier by ScheduleBatch().
  void ProcessBatch(std::unique_ptr<Batch<TaskType>> batch,
                    std::vector<std::unique_ptr<TaskType>> padding_task);

  // Determines whether the queue is empty, i.e. has no tasks waiting or being
  // processed.
  bool IsEmpty() const;

  // Marks the queue closed, and waits until it is empty.
  void CloseAndWaitUntilEmpty();

  bool closed() const TF_NO_THREAD_SAFETY_ANALYSIS { return closed_.load(); }

 private:
  // Computes the max_execution_batch_size of the queue based on queue options.
  static size_t GetMaxExecutionBatchSize(
      const typename SharedBatchScheduler<TaskType>::QueueOptions& options) {
    // If `enable_large_batch_splitting`, returns `max_execution_batch_size`
    // configured by user options directly; returns `input_batch_size_limit`
    // otherwise.
    //
    // Note `input_batch_size_limit` is used for backward compatibitliy ->
    // users may not specify `max_execution_batch_size` explicitly.
    if (options.enable_large_batch_splitting) {
      return options.max_execution_batch_size;
    } else {
      return options.input_batch_size_limit;
    }
  }

  // Same as IsEmpty(), but assumes the caller already holds a lock on 'mu_'.
  bool IsEmptyInternal() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns true iff the task is a low priority task based on the queue option.
  bool IsLowPriorityTask(std::unique_ptr<TaskType>* task);

  // Implementation of Schedule above. Enqueues `task` as it
  // is or split it inline (eagerly) to form batches to be processed by
  // `Queue<TaskType>::ProcessBatch`
  absl::Status ScheduleWithoutOrEagerSplitImpl(std::unique_ptr<TaskType>* task)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Pads the open batch until it is full with low priority tasks.
  void PadOpenBatchWithLowPriorityTasks() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Closes the open batch residing at the back of std::deque, and inserts a
  // fresh open batch behind it.
  void StartNewBatch() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Split `input task` into `output_tasks` according to 'task_sizes'.
  absl::Status SplitInputBatchIntoSubtasks(
      std::unique_ptr<TaskType>* input_task,
      std::vector<std::unique_ptr<TaskType>>* output_tasks)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Determines whether the open batch residing at the back of
  // 'high_priority_batches_' is currently schedulable.
  bool IsOpenBatchSchedulable() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::optional<BatchPriorityKey> PeekBatchPriorityImpl() const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Determines whether the low priority tasks in `low_priority_tasks_` can form
  // a batch on their own. If yes, returns a batch that is ready to be
  // processed. Otherwise, returns an empty unique_ptr.
  std::unique_ptr<Batch<TaskType>> ScheduleLowPriorityBatch()
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Same as SchedulingCapacity(), but assumes the caller already holds a
  // lock on 'mu_'.
  size_t SchedulingCapacityInternal() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns an error if queue doesn't have capacity for this task.
  //
  // `task` must outlive this method.
  absl::Status ValidateBatchTaskQueueCapacity(TaskType* task) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns an error if the low priority task queue doesn't have capacity for
  // this task using the low priority batch options. Since the low priority
  // tasks are not batched until they get scheduled, it only checks that a
  // single task does not it exceed input batch size limit and the total size of
  // the tasks in the queue does not exceed the max batch size * max enqueued
  // batch sizes.
  absl::Status ValidateLowPriorityTaskQueueCapacity(const TaskType& task) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // The task size of the last batch in the queue.
  size_t tail_batch_task_size() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns the number of enqueued batches.
  int64 num_enqueued_batches() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Gets the appropriate batches.
  std::deque<std::unique_ptr<Batch<TaskType>>>& GetBatches()
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Gets the appropriate batches (const version).
  const std::deque<std::unique_ptr<Batch<TaskType>>>& GetBatches() const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Gets the low priority task queue.
  TaskQueue<TaskType>& GetLowPriorityTaskQueue()
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Retrieves the tasks up to the specified size from the low priority task
  // queue. It will immediately return an empty vector when
  // enable_priority_queue is false.
  std::vector<std::unique_ptr<TaskType>> GetLowPriorityTasks(size_t size);

  const typename SharedBatchScheduler<TaskType>::QueueOptions options_;

  // The environment to use.
  Env* env_;

  // The maximum batch size to be executed by `Queue::ProcessBatch`.
  // See the comment of QueueOptions and helper function
  // `GetMaxExecutionBatchSize` for more details on what it means.
  const size_t max_execution_batch_size_;

  // A callback invoked to processes a batch of work units. Always invoked
  // from a batch thread.
  ProcessBatchCallback process_batch_callback_;

  // A callback invoked to notify the scheduler that a new batch has become
  // schedulable.
  SchedulableBatchCallback schedulable_batch_callback_;

  mutable mutex mu_;

  // Whether this queue can accept new tasks. This variable is monotonic: it
  // starts as false, and then at some point gets set to true and remains true
  // for the duration of this object's life.
  std::atomic<bool> closed_ TF_GUARDED_BY(mu_){false};

  // The enqueued tasks for low priority inputs.
  // Each element corresponds to a task to be dequeued. These tasks to be
  // consumed by `Queue<TaskType>::ProcessBatch` to either pad the high priority
  // batches below or form their own batch to be executed.
  TaskQueue<TaskType> low_priority_tasks_ TF_GUARDED_BY(mu_);

  // The enqueued batches for high priority input.
  // Each element corresponds to a task to be dequeued and processed by
  // `Queue<TaskType>::ProcessBatch`.
  std::deque<std::unique_ptr<Batch<TaskType>>> high_priority_batches_
      TF_GUARDED_BY(mu_);

  // The counter of the TraceMe context ids.
  uint64 traceme_context_id_counter_ TF_GUARDED_BY(mu_) = 0;

  // The time at which the first task was added to the open (back-most) batch
  // in 'high_priority_batches_'. Valid iff that batch contains at least one
  // task.
  //
  // Note that when using a batch padding policy other than PAD_UP, this field
  // might contain an approximate value.
  uint64 open_batch_start_time_micros_ TF_GUARDED_BY(mu_);

  // Whether this queue contains a batch that is eligible to be scheduled.
  // Used to keep track of when to call 'schedulable_batch_callback_'.
  bool schedulable_batch_ TF_GUARDED_BY(mu_) = false;

  // The number of batches currently being processed by batch threads.
  // Incremented in ScheduleBatch() and decremented in ProcessBatch().
  int num_batches_being_processed_ TF_GUARDED_BY(mu_) = 0;

  // Used by CloseAndWaitUntilEmpty() to wait until the queue is empty, for
  // the case in which the queue is not empty when CloseAndWaitUntilEmpty()
  // starts. When ProcessBatch() dequeues the last batch and makes the queue
  // empty, if 'empty_notification_' is non-null it calls
  // 'empty_notification_->Notify()'.
  Notification* empty_notification_ TF_GUARDED_BY(mu_) = nullptr;

  Queue(const Queue&) = delete;
  void operator=(const Queue&) = delete;
};

// A RAII-style object that points to a Queue and implements
// the BatchScheduler API. To be handed out to clients who call AddQueue().
template <typename TaskType>
class QueueHandle : public BatchScheduler<TaskType> {
 public:
  QueueHandle(std::shared_ptr<SharedBatchScheduler<TaskType>> scheduler,
              Queue<TaskType>* queue);
  ~QueueHandle() override;

  absl::Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;

  size_t max_task_size() const override { return queue_->max_task_size(); }

 private:
  // The scheduler that owns 'queue_'.
  std::shared_ptr<SharedBatchScheduler<TaskType>> scheduler_;

  // The queue this handle wraps. Owned by 'scheduler_', which keeps it alive at
  // least until this class's destructor closes it.
  Queue<TaskType>* queue_;

  QueueHandle(const QueueHandle&) = delete;
  void operator=(const QueueHandle&) = delete;
};

}  // namespace internal

template <typename TaskType>
absl::Status SharedBatchScheduler<TaskType>::Create(
    const Options& options,
    std::shared_ptr<SharedBatchScheduler<TaskType>>* scheduler) {
  if (options.num_batch_threads < 1) {
    return errors::InvalidArgument("num_batch_threads must be positive; was ",
                                   options.num_batch_threads);
  }

  if (options.use_global_scheduler) {
    static std::shared_ptr<SharedBatchScheduler<TaskType>>* global_scheduler =
        [&]() {
          return new std::shared_ptr<SharedBatchScheduler<TaskType>>(
              new SharedBatchScheduler<TaskType>(options));
        }();
    *scheduler = *global_scheduler;
    return absl::OkStatus();
  }

  scheduler->reset(new SharedBatchScheduler<TaskType>(options));
  return absl::OkStatus();
}

template <typename TaskType>
SharedBatchScheduler<TaskType>::~SharedBatchScheduler() {
  // Wait until the batch threads finish clearing out and deleting the closed
  // queues.
  for (;;) {
    {
      mutex_lock l(mu_);
      if (queues_.empty()) {
        break;
      }
    }
    const int64_t kSleepTimeMicros = 100;
    options_.env->SleepForMicroseconds(kSleepTimeMicros);
  }
  // Delete the batch threads before allowing state the threads may access (e.g.
  // 'mu_') to be deleted.
  batch_threads_.clear();
}

template <typename TaskType>
absl::Status SharedBatchScheduler<TaskType>::AddQueue(
    const QueueOptions& options, ProcessBatchCallback process_batch_callback,
    std::unique_ptr<BatchScheduler<TaskType>>* queue) {
  QueueOptions rewrite_options = options;
  if ((!rewrite_options.enable_large_batch_splitting) &&
      rewrite_options.max_enqueued_batches == 0) {
    // Many existing models (with very low QPS) rely on this option to be >0.
    // Rewrite and set this to one and retain old behavior to allow such models
    // to continue to work.
    //
    // Note, technically an invalid-argument error should be returned, but
    // that may break such models.
    rewrite_options.max_enqueued_batches = 1;
  }
  return AddQueueAfterRewritingOptions(rewrite_options, process_batch_callback,
                                       queue);
}

template <typename TaskType>
absl::Status SharedBatchScheduler<TaskType>::AddQueueAfterRewritingOptions(
    const QueueOptions& options, ProcessBatchCallback process_batch_callback,
    std::unique_ptr<BatchScheduler<TaskType>>* queue) {
  if (options.input_batch_size_limit == 0) {
    return errors::InvalidArgument(
        "input_batch_size_limit must be positive; was ",
        options.input_batch_size_limit);
  }
  if (options.batch_timeout_micros < 0) {
    return errors::InvalidArgument(
        "batch_timeout_micros must be non-negative; was ",
        options.batch_timeout_micros);
  }
  if (options.max_enqueued_batches == 0) {
    return errors::InvalidArgument(
        "max_enqueued_batches must be positive; was ",
        options.max_enqueued_batches);
  }

  if (options.enable_large_batch_splitting &&
      options.split_input_task_func == nullptr) {
    return errors::InvalidArgument(
        "split_input_task_func must be specified when split_input_task is "
        "true: ",
        options.enable_large_batch_splitting);
  }

  if (options.enable_large_batch_splitting &&
      (options.input_batch_size_limit < options.max_execution_batch_size)) {
    return errors::InvalidArgument(
        "When enable_large_batch_splitting is true, input_batch_size_limit "
        "must be "
        "greater than or equal to max_execution_batch_size.",
        options.enable_large_batch_splitting, options.input_batch_size_limit,
        options.max_execution_batch_size);
  }

  auto schedulable_batch_callback = [this] {
    mutex_lock l(mu_);
    schedulable_batch_cv_.notify_one();
  };
  auto internal_queue =
      std::unique_ptr<internal::Queue<TaskType>>(new internal::Queue<TaskType>(
          options, options_.env, process_batch_callback,
          schedulable_batch_callback));
  auto handle = std::unique_ptr<BatchScheduler<TaskType>>(
      new internal::QueueHandle<TaskType>(this->shared_from_this(),
                                          internal_queue.get()));
  {
    mutex_lock l(mu_);
    queues_.push_back(std::move(internal_queue));
    if (next_queue_to_schedule_ == queues_.end()) {
      next_queue_to_schedule_ = queues_.begin();
    }
  }
  *queue = std::move(handle);
  return absl::OkStatus();
}

template <typename TaskType>
SharedBatchScheduler<TaskType>::SharedBatchScheduler(const Options& options)
    : options_(options), next_queue_to_schedule_(queues_.end()) {
  // Kick off the batch threads.
  PeriodicFunction::Options periodic_fn_options;
  periodic_fn_options.thread_name_prefix =
      strings::StrCat(options.thread_pool_name, "_");
  for (int i = 0; i < options.num_batch_threads; ++i) {
    std::unique_ptr<PeriodicFunction> thread(new PeriodicFunction(
        [this] { this->ThreadLogic(); },
        0 /* function invocation interval time */, periodic_fn_options));
    batch_threads_.push_back(std::move(thread));
  }
}

template <typename TaskType>
bool SharedBatchScheduler<TaskType>::BatchExists(
    const BatchTaskUniquePtr& batch_to_process) {
  return batch_to_process != nullptr;
}

template <typename TaskType>
void SharedBatchScheduler<TaskType>::GetNextWorkItem_Locked(
    internal::Queue<TaskType>** queue_for_batch_out,
    BatchTaskUniquePtr* batch_to_process_out) {
  BatchTaskUniquePtr batch_to_process;
  internal::Queue<TaskType>* queue_for_batch = nullptr;
  std::optional<typename internal::Queue<TaskType>::BatchPriorityKey>
      batch_priority_key;
  const int num_queues = queues_.size();
  for (int num_queues_tried = 0;
       !BatchExists(batch_to_process) && num_queues_tried < num_queues;
       ++num_queues_tried) {
    DCHECK(next_queue_to_schedule_ != queues_.end());

    // If a closed queue responds to ScheduleBatch() with nullptr, the queue
    // will never yield any further batches so we can drop it. To avoid a
    // race, we take a snapshot of the queue's closedness state *before*
    // calling ScheduleBatch().
    const bool queue_closed = (*next_queue_to_schedule_)->closed();

    bool queue_has_work = false;

    if (options_.rank_queues) {
      auto key = (*next_queue_to_schedule_)->PeekBatchPriority();
      queue_has_work = key.has_value();
      if (key.has_value() && (!batch_priority_key.has_value() ||
                              key.value() < batch_priority_key.value())) {
        batch_priority_key = key;
        queue_for_batch = next_queue_to_schedule_->get();
      }
    } else {
      // Ask '*next_queue_to_schedule_' if it wants us to process a batch.
      batch_to_process = (*next_queue_to_schedule_)->ScheduleBatch();
      queue_has_work = BatchExists(batch_to_process);

      if (queue_has_work) {
        queue_for_batch = next_queue_to_schedule_->get();
      }
    }

    // Advance 'next_queue_to_schedule_'.
    if (queue_closed && (*next_queue_to_schedule_)->IsEmpty() &&
        !queue_has_work) {
      // We've encountered a closed queue with no work to do. Drop it.
      DCHECK_NE(queue_for_batch, next_queue_to_schedule_->get());
      next_queue_to_schedule_ = queues_.erase(next_queue_to_schedule_);
    } else {
      ++next_queue_to_schedule_;
    }
    if (next_queue_to_schedule_ == queues_.end() && !queues_.empty()) {
      // We've hit the end. Wrap to the first queue.
      next_queue_to_schedule_ = queues_.begin();
    }
  }

  if (options_.rank_queues && batch_priority_key.has_value()) {
    batch_to_process = queue_for_batch->ScheduleBatch();
  }

  *queue_for_batch_out = queue_for_batch;
  *batch_to_process_out = std::move(batch_to_process);
}

template <typename TaskType>
void SharedBatchScheduler<TaskType>::ThreadLogic() {
  // A batch to process next (or nullptr if no work to do).
  BatchTaskUniquePtr batch_to_process;
  // The queue with which 'batch_to_process' is associated.
  internal::Queue<TaskType>* queue_for_batch = nullptr;
  {
    mutex_lock l(mu_);
    while (true) {
      GetNextWorkItem_Locked(&queue_for_batch, &batch_to_process);
      if (BatchExists(batch_to_process)) break;
      // We couldn't find any work to do. Wait until a new batch becomes
      // schedulable, or some time has elapsed, before checking again.
      const int64_t kTimeoutMillis =
          1;  // The smallest accepted granule of time.
      WaitForMilliseconds(&l, &schedulable_batch_cv_, kTimeoutMillis);
      if (queues_.empty()) return;
    }
  }

  size_t batch_size_to_schedule = batch_to_process->size();
  queue_for_batch->ProcessBatch(
      std::move(batch_to_process),
      queue_for_batch->GetLowPriorityTasksForPadding(batch_size_to_schedule));
}

namespace internal {

template <typename TaskType>
Queue<TaskType>::Queue(
    const typename SharedBatchScheduler<TaskType>::QueueOptions& options,
    Env* env, ProcessBatchCallback process_batch_callback,
    SchedulableBatchCallback schedulable_batch_callback)
    : options_(options),
      env_(env),
      max_execution_batch_size_(GetMaxExecutionBatchSize(options_)),
      process_batch_callback_(process_batch_callback),
      schedulable_batch_callback_(schedulable_batch_callback) {
  // Set the higher 32 bits of traceme_context_id_counter_ to be the creation
  // time of the queue. This prevents the batches in different queues to have
  // the same traceme_context_id_counter_.
  traceme_context_id_counter_ = (absl::GetCurrentTimeNanos() & 0xFFFFFFFF)
                                << 32;
  GetBatches().emplace_back(new Batch<TaskType>);
}

template <typename TaskType>
Queue<TaskType>::~Queue() {
  mutex_lock l(mu_);
  DCHECK(IsEmptyInternal());
  GetBatches().back()->Close();
}

template <typename TaskType>
bool Queue<TaskType>::IsLowPriorityTask(std::unique_ptr<TaskType>* task) {
  if (!options_.enable_priority_queue) {
    return false;
  }

  // The criticality is defined only when the task is a derived class of
  // BatchTask.
  if constexpr (std::is_base_of_v<BatchTask, TaskType>) {
    // TODO(b/316379576): Make the criticality and priority configurable.
    return ((*task)->criticality() ==
                tsl::criticality::Criticality::kSheddablePlus ||
            (*task)->criticality() ==
                tsl::criticality::Criticality::kSheddable);
  }

  // Otherwise, consider it a high priority task and return false.
  return false;
}

template <typename TaskType>
absl::Status Queue<TaskType>::ScheduleWithoutOrEagerSplitImpl(
    std::unique_ptr<TaskType>* task) {
  // TODO(b/161857471):
  // Add test coverage when when concurrent incoming batches arrives and
  // use up all queue capacity.
  TF_RETURN_IF_ERROR(ValidateBatchTaskQueueCapacity((*task).get()));

  std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();

  const int64_t open_batch_remaining_slot =
      max_execution_batch_size() - batches.back()->size();

  const int64_t input_task_size = (*task)->size();

  std::vector<std::unique_ptr<TaskType>> output_tasks;

  if (input_task_size <= open_batch_remaining_slot ||
      !options_.enable_large_batch_splitting) {
    // This is the fast path when input doesn't need to be split.
    output_tasks.push_back(std::move(*task));
  } else {
    TF_RETURN_IF_ERROR(SplitInputBatchIntoSubtasks(task, &output_tasks));
  }

  for (int i = 0; i < output_tasks.size(); ++i) {
    if (batches.back()->size() + output_tasks[i]->size() >
        max_execution_batch_size()) {
      StartNewBatch();
    }
    if (batches.back()->empty()) {
      open_batch_start_time_micros_ = env_->NowMicros();
    }
    tsl::profiler::TraceMeProducer trace_me(
        [&output_tasks, i] {
          return profiler::TraceMeEncode("ScheduleOutputTask",
                                         {{"size", output_tasks[i]->size()}});
        },
        tsl::profiler::ContextType::kSharedBatchScheduler,
        batches.back()->traceme_context_id());
    batches.back()->AddTask(std::move(output_tasks[i]), env_->NowMicros());
  }

  return absl::OkStatus();
}

template <typename TaskType>
void Queue<TaskType>::PadOpenBatchWithLowPriorityTasks() {
  std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();

  const bool should_pad = options_.enable_priority_queue &&
                          options_.mixed_priority_batching_policy ==
                              MixedPriorityBatchingPolicy::kPriorityMerge &&
                          batches.size() == 1 && IsOpenBatchSchedulable();
  if (!should_pad) {
    return;
  }

  // If true, the next low priority task couldn't fit in the remaining space of
  // the open batch.
  bool out_of_space = false;

  while (!low_priority_tasks_.empty() && !out_of_space) {
    const int64_t open_batch_remaining_slot =
        max_execution_batch_size() - batches.back()->size();
    if (open_batch_remaining_slot <= 0) {
      // Terminate early if the open batch is full. Remaining low priority tasks
      // will be re-checked during the next batch formation opportunity.
      return;
    }

    uint64 task_time = low_priority_tasks_.EarliestTaskStartTime().value();
    std::unique_ptr<TaskType> task = low_priority_tasks_.RemoveTask();

    const int64_t input_task_size = task->size();

    std::vector<std::unique_ptr<TaskType>> output_tasks;

    if (input_task_size <= open_batch_remaining_slot ||
        !options_.enable_large_batch_splitting) {
      // This is the fast path when input doesn't need to be split.
      output_tasks.push_back(std::move(task));
    } else {
      absl::Status status = SplitInputBatchIntoSubtasks(&task, &output_tasks);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to split low priority task: " << status;
        continue;
      }
    }

    for (int i = 0; i < output_tasks.size(); ++i) {
      if (batches.back()->size() + output_tasks[i]->size() >
          max_execution_batch_size()) {
        low_priority_tasks_.PrependTask(std::move(output_tasks[i]), task_time);
        out_of_space = true;
        // NOTE: Future iterations of this loop will also hit this case but are
        // needed to re-add all the unused tasks to the low priority queue.
        continue;
      }

      if (batches.back()->empty()) {
        open_batch_start_time_micros_ = task_time;
      } else {
        open_batch_start_time_micros_ =
            std::min(open_batch_start_time_micros_, task_time);
      }

      tsl::profiler::TraceMeProducer trace_me(
          [&output_tasks, i] {
            return profiler::TraceMeEncode("ScheduleOutputTask",
                                           {{"size", output_tasks[i]->size()}});
          },
          tsl::profiler::ContextType::kSharedBatchScheduler,
          batches.back()->traceme_context_id());

      batches.back()->AddTask(std::move(output_tasks[i]));
    }
  }
}

template <typename TaskType>
absl::Status Queue<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  const bool large_batch_splitting = options_.enable_large_batch_splitting;
  tsl::profiler::TraceMe trace_me([task, large_batch_splitting] {
    return profiler::TraceMeEncode(
        large_batch_splitting ? "ScheduleWithEagerSplit"
                              : "ScheduleWithoutSplit",
        {{"batching_input_task_size", (*task)->size()}});
  });

  bool notify_of_schedulable_batch = false;
  {
    mutex_lock l(mu_);

    DCHECK(!closed_);

    if (IsLowPriorityTask(task)) {
      // Insert the task to the low priority task queue instead of the high
      // priority batch queue below.
      TF_RETURN_IF_ERROR(ValidateLowPriorityTaskQueueCapacity(**task));
      low_priority_tasks_.AddTask(std::move(*task), env_->NowMicros());
    } else {
      TF_RETURN_IF_ERROR(ScheduleWithoutOrEagerSplitImpl(task));
    }

    // Check if the batch queue has a schedulable batch and mark it schedulable
    // if it not already marked.
    if (!schedulable_batch_) {
      if (GetBatches().size() > 1 || IsOpenBatchSchedulable()) {
        schedulable_batch_ = true;
        notify_of_schedulable_batch = true;
      }
    }
  }

  if (notify_of_schedulable_batch) {
    schedulable_batch_callback_();
  }

  return absl::OkStatus();
}

template <typename TaskType>
size_t Queue<TaskType>::NumEnqueuedTasks() const {
  size_t num_enqueued_tasks = 0;
  mutex_lock l(mu_);
  for (const auto& batch : GetBatches()) {
    num_enqueued_tasks += batch->num_tasks();
  }
  return num_enqueued_tasks + low_priority_tasks_.num_tasks();
}

template <typename TaskType>
size_t Queue<TaskType>::SchedulingCapacity() const {
  mutex_lock l(mu_);
  return SchedulingCapacityInternal();
}

template <typename TaskType>
size_t Queue<TaskType>::SchedulingCapacityInternal() const {
  const int64 num_new_batches_schedulable =
      static_cast<int64_t>(options_.max_enqueued_batches) -
      this->num_enqueued_batches();
  const int64 execution_batch_size_limit = max_execution_batch_size();
  const int64 open_batch_capacity =
      execution_batch_size_limit - this->tail_batch_task_size();
  // Note the returned value is guaranteed to be not negative, since
  // enqueue operation could only happen if queue has enough capacity.
  return (num_new_batches_schedulable * execution_batch_size_limit) +
         open_batch_capacity;
}

template <typename TaskType>
absl::Status Queue<TaskType>::ValidateBatchTaskQueueCapacity(
    TaskType* task) const {
  // Check if the task size is larger than the batch size limit, regardless of
  // the batch capacity.
  if (task->size() > options_.input_batch_size_limit) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Task size %d is larger than maximum input batch size %d", task->size(),
        options_.input_batch_size_limit));
  }

  if (options_.enable_large_batch_splitting) {
    if (task->size() > SchedulingCapacityInternal()) {
      return errors::Unavailable(
          "The batch scheduling queue to which this task was submitted is "
          "full; task size is ",
          task->size(), " but scheduling capacity is only ",
          SchedulingCapacityInternal(),
          " (num_enqueued_batches=", num_enqueued_batches(),
          ", max_enqueued_batches=", options_.max_enqueued_batches,
          ", open_batch_size=", tail_batch_task_size(),
          ", max_execution_batch_size=", max_execution_batch_size(), ")");
    }
    return absl::OkStatus();
  }

  // NOTE, the capacity checking below is loose and is retained
  // for backward compatibility that was broken due to the merge of no-split
  // and eager split.
  // There are existing clients/models that rely on the loose check
  // and can get errors after the merge. Retaining the old behavior
  // allows such models to continue to work.
  //
  // We need to revisit/remove this check after we fix model configs.
  const std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
  if (batches.back()->size() + task->size() > options_.input_batch_size_limit) {
    if (batches.size() >= options_.max_enqueued_batches) {
      return errors::Unavailable(
          "The batch scheduling queue to which this task was submitted is "
          "full; currently ",
          batches.size(), " batches enqueued and max_enqueued_batches is ",
          options_.max_enqueued_batches);
    }
  }
  return absl::OkStatus();
}

template <typename TaskType>
absl::Status Queue<TaskType>::ValidateLowPriorityTaskQueueCapacity(
    const TaskType& task) const {
  // Unlike the high priority batch capacity validation where having only
  // input_batch_size_limit without max_execution_batch_size is allowed, it
  // doesn't have the backward compatibility check and always assume that
  // max_execution_batch_size is present.
  if (task.size() >
      options_.low_priority_queue_options.max_execution_batch_size) {
    return absl::UnavailableError(absl::StrFormat(
        "The low priority task queue to which this task was submitted has "
        "max_execution_batch_size=%d and the task size is %d",
        options_.low_priority_queue_options.max_execution_batch_size,
        task.size()));
  }
  if (low_priority_tasks_.size() + task.size() >
      options_.low_priority_queue_options.max_enqueued_batches *
          options_.low_priority_queue_options.max_execution_batch_size) {
    return absl::UnavailableError(absl::StrFormat(
        "The low priority task queue to which this task was submitted does not "
        "have the capcity to handle this task; currently the low priority "
        "queue has %d tasks enqueued and the submitted task size is %d while "
        "max_enqueued_batches=%d and max_execution_batch_size=%d",
        low_priority_tasks_.size(), task.size(),
        options_.low_priority_queue_options.max_enqueued_batches,
        options_.low_priority_queue_options.max_execution_batch_size));
  }
  return absl::OkStatus();
}

template <typename TaskType>
typename SharedBatchScheduler<TaskType>::BatchTaskUniquePtr
Queue<TaskType>::ScheduleBatch() {
  // The batch to schedule, which we may populate below. (If left as nullptr,
  // that means we are electing not to schedule a batch at this time.)
  std::unique_ptr<Batch<TaskType>> batch_to_schedule;

  {
    mutex_lock l(mu_);

    std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();

    // Just in time merging of low priority tasks into the open batch.
    PadOpenBatchWithLowPriorityTasks();

    // Consider closing the open batch at this time, to schedule it.
    if (batches.size() == 1 && IsOpenBatchSchedulable()) {
      // Support BatchPaddingPolicy::kBatchDown and
      // BatchPaddingPolicy::kMinimizeTpuCostPerRequest. We do this before
      // starting a new batch because starting a new batch will close the old
      // batch, making it read-only.
      Batch<TaskType>& old_batch = *batches[0];
      uint64 old_batch_time = old_batch.EarliestTaskStartTime().value();
      std::vector<std::unique_ptr<TaskType>> trimmed_tasks;
      MaybeBatchDown(
          /* batch= */ old_batch,
          /* allowed_batch_sizes= */ options_.allowed_batch_sizes,
          /* disable_padding= */ options_.disable_padding,
          /* batch_padding_policy= */ options_.batch_padding_policy,
          /* model_batch_stats= */ options_.model_batch_stats,
          /* out_trimmed_tasks= */ trimmed_tasks);

      StartNewBatch();

      // Move the trimmed tasks, if any, into the new batch.
      Batch<TaskType>& new_batch = *batches[1];
      for (std::unique_ptr<TaskType>& task : trimmed_tasks) {
        new_batch.AddTask(std::move(task), old_batch_time);
      }
      if (!new_batch.empty()) {
        // TODO - b/325954758: Reconsider the starting time of a trimmed batch.
        //
        // Ideally, we'd set open_batch_start_time_micros_ to time we received
        // the first task in the open batch, but we don't have this information
        // here. For now, we're trying as alternative solution that doesn't
        // require adding time to each task: assume that requests arrived at a
        // steady rate and therefore use a point between the old value of
        // open_batch_start_time_micros_ and NOW.
        //
        // Let's say that originally, the batch had 10 requests, and we want to
        // schedule a batch of size 8 and leave 2 requests in the open batch
        // (new_batch). Then, variable `position` is 0.8, which means we have to
        // set open_batch_start_time_micros_ to be at a position of 80% between
        // open_batch_start_time_micros_ and now.
        double position = static_cast<double>(old_batch.size()) /
                          (old_batch.size() + new_batch.size());
        open_batch_start_time_micros_ +=
            (env_->NowMicros() - open_batch_start_time_micros_) * position;
      }
    }

    if (batches.size() >= 2) {
      // There is at least one closed batch that is ready to be scheduled.
      batch_to_schedule = std::move(batches.front());
      batches.pop_front();
    }

    if (batch_to_schedule == nullptr) {
      // If there was no schedulable batch in the batch queue, try to schedule
      // from the low priority task queue.
      batch_to_schedule = ScheduleLowPriorityBatch();
    }

    if (batch_to_schedule == nullptr) {
      // There is neither high nor low priority batch that can be scheduled,
      // mark the condition false and return the nullptr.
      schedulable_batch_ = false;
      return batch_to_schedule;
    }

    // Otherwise, increment the counter and return the batch.
    ++num_batches_being_processed_;
  }
  return batch_to_schedule;
}

template <typename TaskType>
std::vector<std::unique_ptr<TaskType>> Queue<TaskType>::GetLowPriorityTasks(
    size_t size) {
  std::vector<std::unique_ptr<TaskType>> low_priority_tasks_to_pad;
  // If priority queue is not enabled, immediately return instead of attempting
  // to acquire a lock.
  if (!options_.enable_priority_queue || size == 0)
    return low_priority_tasks_to_pad;
  {
    mutex_lock l(mu_);
    low_priority_tasks_to_pad = GetLowPriorityTaskQueue().RemoveTask(size);
  }
  return low_priority_tasks_to_pad;
}

template <typename TaskType>
std::vector<std::unique_ptr<TaskType>>
Queue<TaskType>::GetLowPriorityTasksForPadding(size_t batch_size) {
  size_t target_batch_size;
  switch (options_.mixed_priority_batching_policy) {
    case MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize:
      target_batch_size = max_execution_batch_size();
      break;
    case MixedPriorityBatchingPolicy::
        kLowPriorityPaddingWithNextAllowedBatchSize:
      target_batch_size = GetNextAllowedBatchSize(
          batch_size, options_.allowed_batch_sizes, options_.disable_padding);
      break;
    default:
      target_batch_size = 0;
      break;
  }

  if (target_batch_size <= batch_size) {
    return {};
  }
  return GetLowPriorityTasks(target_batch_size - batch_size);
}

template <typename TaskType>
void Queue<TaskType>::ProcessBatch(
    std::unique_ptr<Batch<TaskType>> batch,
    std::vector<std::unique_ptr<TaskType>> padding_task) {
  tsl::profiler::TraceMeConsumer trace_me(
      [&] {
        return profiler::TraceMeEncode(
            "ProcessBatch", {{"batch_size_before_padding", batch->size()},
                             {"_r", 2} /*root_event*/});
      },
      tsl::profiler::ContextType::kSharedBatchScheduler,
      batch->traceme_context_id());

  if (std::holds_alternative<ProcessBatchCallbackWithoutPaddingTasks>(
          process_batch_callback_)) {
    std::get<ProcessBatchCallbackWithoutPaddingTasks>(process_batch_callback_)(
        std::move(batch));
  } else {
    std::get<ProcessBatchCallbackWithPaddingTasks>(process_batch_callback_)(
        std::move(batch), std::move(padding_task));
  }

  {
    mutex_lock l(mu_);
    --num_batches_being_processed_;
    if (empty_notification_ != nullptr && IsEmptyInternal()) {
      empty_notification_->Notify();
    }
  }
}

template <typename TaskType>
bool Queue<TaskType>::IsEmpty() const {
  mutex_lock l(mu_);
  return IsEmptyInternal();
}

template <typename TaskType>
void Queue<TaskType>::CloseAndWaitUntilEmpty() {
  Notification empty;
  {
    mutex_lock l(mu_);
    closed_ = true;
    if (IsEmptyInternal()) {
      empty.Notify();
    } else {
      // Arrange for ProcessBatch() to notify when the queue becomes empty.
      empty_notification_ = &empty;
    }
  }
  empty.WaitForNotification();
}

template <typename TaskType>
bool Queue<TaskType>::IsEmptyInternal() const {
  const std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
  return num_batches_being_processed_ == 0 && batches.size() == 1 &&
         batches.back()->empty() && low_priority_tasks_.empty();
}

template <typename TaskType>
void Queue<TaskType>::StartNewBatch() {
  std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
  batches.back()->Close();
  batches.emplace_back(new Batch<TaskType>(++traceme_context_id_counter_));
}

template <typename TaskType>
absl::Status Queue<TaskType>::SplitInputBatchIntoSubtasks(
    std::unique_ptr<TaskType>* input_task,
    std::vector<std::unique_ptr<TaskType>>* output_tasks) {
  const int open_batch_remaining_slot =
      max_execution_batch_size() - this->tail_batch_task_size();
  return options_.split_input_task_func(
      std::move(input_task), open_batch_remaining_slot,
      max_execution_batch_size(), std::move(output_tasks));
}

template <typename TaskType>
bool Queue<TaskType>::IsOpenBatchSchedulable() const {
  return PeekBatchPriorityImpl().has_value();
}

template <typename TaskType>
std::optional<typename Queue<TaskType>::BatchPriorityKey>
Queue<TaskType>::PeekBatchPriority() const {
  {
    mutex_lock l(mu_);
    return PeekBatchPriorityImpl();
  }
}

template <typename TaskType>
std::optional<typename Queue<TaskType>::BatchPriorityKey>
Queue<TaskType>::PeekBatchPriorityImpl() const {
  const int kHighPriority = 1;
  const int kLowPriority = 2;

  const std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();

  if (batches.size() >= 2) {
    Batch<TaskType>* batch = batches.front().get();
    return std::make_pair(kHighPriority,
                          batch->EarliestTaskStartTime().value());
  }

  Batch<TaskType>* open_batch = batches.back().get();

  size_t effective_batch_size = open_batch->size();
  uint64 effective_start_time_micros = open_batch_start_time_micros_;
  int64_t effective_batch_timeout_micros = options_.batch_timeout_micros;
  if (effective_batch_size == 0) {
    // open_batch_start_time_micros_ is not valid for an empty batch.
    effective_start_time_micros = env_->NowMicros();
  }

  if (options_.enable_priority_queue &&
      options_.mixed_priority_batching_policy ==
          MixedPriorityBatchingPolicy::kPriorityMerge) {
    if (effective_batch_size == 0) {
      effective_batch_timeout_micros =
          options_.low_priority_queue_options.batch_timeout_micros;
    }

    effective_batch_size += low_priority_tasks_.size();

    auto low_priority_earliest_start_time =
        low_priority_tasks_.EarliestTaskStartTime();
    if (low_priority_earliest_start_time.has_value()) {
      effective_start_time_micros = std::min(effective_start_time_micros,
                                             *low_priority_earliest_start_time);
    }
  }

  if (effective_batch_size == 0) {
    return std::nullopt;
  }

  bool schedulable = closed_ ||
                     effective_batch_size >= max_execution_batch_size() ||
                     env_->NowMicros() >= effective_start_time_micros +
                                              effective_batch_timeout_micros;

  if (!schedulable) {
    return std::nullopt;
  }

  int priority = open_batch->empty() ? kLowPriority : kHighPriority;
  return std::make_pair(priority, effective_start_time_micros);
}

template <typename TaskType>
std::unique_ptr<Batch<TaskType>> Queue<TaskType>::ScheduleLowPriorityBatch() {
  std::unique_ptr<Batch<TaskType>> batch_to_schedule;
  if (!options_.enable_priority_queue || low_priority_tasks_.empty() ||
      options_.mixed_priority_batching_policy ==
          MixedPriorityBatchingPolicy::kPriorityMerge) {
    // Return early if priority queue is disabled or there is no low priority
    // task. Note that the priority_merge policy does all scheduling in
    // ScheduleBatch().
    return batch_to_schedule;
  }
  if (env_->NowMicros() <
          *low_priority_tasks_.EarliestTaskStartTime() +
              options_.low_priority_queue_options.batch_timeout_micros &&
      low_priority_tasks_.size() <
          options_.low_priority_queue_options.max_execution_batch_size) {
    // Return early if the low priority tasks can't fill up the max batch size
    // and the earliest task didn't time out.
    return batch_to_schedule;
  }
  if (!GetBatches().empty() && !GetBatches().front()->empty()) {
    // Return early if there is a non-empty high priority batch in the queue.
    return batch_to_schedule;
  }

  batch_to_schedule = std::make_unique<Batch<TaskType>>();
  for (std::unique_ptr<TaskType>& task : low_priority_tasks_.RemoveTask(
           options_.low_priority_queue_options.max_execution_batch_size)) {
    batch_to_schedule->AddTask(std::move(task), env_->NowMicros());
  }
  batch_to_schedule->Close();

  return batch_to_schedule;
}

template <typename TaskType>
size_t Queue<TaskType>::tail_batch_task_size() const {
  return GetBatches().back()->size();
}

template <typename TaskType>
int64 Queue<TaskType>::num_enqueued_batches() const {
  return GetBatches().size();
}

template <typename TaskType>
std::deque<std::unique_ptr<Batch<TaskType>>>& Queue<TaskType>::GetBatches() {
  return high_priority_batches_;
}

template <typename TaskType>
const std::deque<std::unique_ptr<Batch<TaskType>>>&
Queue<TaskType>::GetBatches() const {
  return high_priority_batches_;
}

template <typename TaskType>
TaskQueue<TaskType>& Queue<TaskType>::GetLowPriorityTaskQueue() {
  return low_priority_tasks_;
}

template <typename TaskType>
QueueHandle<TaskType>::QueueHandle(
    std::shared_ptr<SharedBatchScheduler<TaskType>> scheduler,
    Queue<TaskType>* queue)
    : scheduler_(scheduler), queue_(queue) {}

template <typename TaskType>
QueueHandle<TaskType>::~QueueHandle() {
  queue_->CloseAndWaitUntilEmpty();
}

template <typename TaskType>
absl::Status QueueHandle<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  return queue_->Schedule(task);
}

template <typename TaskType>
size_t QueueHandle<TaskType>::NumEnqueuedTasks() const {
  return queue_->NumEnqueuedTasks();
}

template <typename TaskType>
size_t QueueHandle<TaskType>::SchedulingCapacity() const {
  return queue_->SchedulingCapacity();
}

}  // namespace internal

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_SHARED_BATCH_SCHEDULER_H_
