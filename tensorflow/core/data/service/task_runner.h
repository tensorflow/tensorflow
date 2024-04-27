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

#include <memory>
#include <optional>
#include <vector>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/cross_trainer_cache.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/thread_safe_buffer.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

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
  // Reports the cardinality of the dataset that created this iterator.
  virtual int64_t Cardinality() const = 0;

  // Saves a checkpoint of the iterator. Returns Tensors that can be called with
  // `Restore()`.
  virtual StatusOr<std::vector<Tensor>> Save() {
    return errors::Unimplemented(
        "Serializing a tf.data service task iterator is unsupported.");
  }

  // Restores the iterator from a checkpoint. `saved_iterator` is the serialized
  // iterator saved by calling `Save()`.
  virtual Status Restore(const std::vector<Tensor>& saved_iterator) {
    return errors::Unimplemented(
        "Restoring from a tf.data service task iterator is unsupported.");
  }

  // Returns the dataset model for performance analysis.
  virtual std::shared_ptr<model::Model> model() const { return nullptr; }
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
  int64_t Cardinality() const override;
  StatusOr<std::vector<Tensor>> Save() override;
  Status Restore(const std::vector<Tensor>& saved_iterator) override;
  std::shared_ptr<model::Model> model() const override;

 private:
  std::unique_ptr<standalone::Dataset> dataset_;
  std::unique_ptr<standalone::Iterator> iterator_;
};

// Interface for providing elements to task consumers.
class TaskRunner {
 public:
  // Creates a `TaskRunner` and stores it in `out`.
  static Status Create(const experimental::WorkerConfig& worker_config,
                       const TaskDef& task_def,
                       std::unique_ptr<TaskIterator> iterator,
                       std::unique_ptr<TaskRunner>& out);
  virtual ~TaskRunner() = default;
  // Gets the next element for the given request.
  virtual Status GetNext(const GetElementRequest& req,
                         GetElementResult& result) = 0;
  // Cancels in-progress `GetNext` requests.
  virtual void Cancel() = 0;
  // Returns the dataset model for performance analysis.
  virtual std::shared_ptr<model::Model> model() const = 0;
};

// A task runner which provides elements on a first-come first-served basis.
// It does not consider which consumer is making the request.
class FirstComeFirstServedTaskRunner : public TaskRunner {
 public:
  explicit FirstComeFirstServedTaskRunner(
      std::unique_ptr<TaskIterator> iterator);
  ~FirstComeFirstServedTaskRunner() override;

  // Gets the next element. It may block if the element is not ready yet.
  Status GetNext(const GetElementRequest& req,
                 GetElementResult& result) override;
  Status GetNext(GetElementResult& result);

  void Cancel() override;

  std::shared_ptr<model::Model> model() const override;

 private:
  // Function to continually prefetch the next element. Returns an error if the
  // task has been cancelled.
  Status PrefetchFn();

  // Runs `PrefetchFn` on a dedicated thread.
  void RunPrefetchThread();

  // Gets the next element from the input iterator.
  StatusOr<GetElementResult> GetNextFromInputIterator() TF_LOCKS_EXCLUDED(mu_);

  const std::shared_ptr<model::Model> model_;
  mutex mu_;
  std::unique_ptr<TaskIterator> iterator_ TF_GUARDED_BY(mu_);
  int64_t element_index_ TF_GUARDED_BY(mu_) = 0;

  ThreadSafeBuffer<GetElementResult> buffer_;
  std::unique_ptr<Thread> prefetch_thread_;

  FirstComeFirstServedTaskRunner(const FirstComeFirstServedTaskRunner&) =
      delete;
  void operator=(const FirstComeFirstServedTaskRunner&) = delete;
};

// A task runner which prefetches elements on a first-come first-served basis
// and caches elements in a sliding-window `CrossTrainerCache`. The cache has a
// bounded size and progresses when a trainer that has consumed all elements in
// the cache. Trainers read from a sliding window of the dataset and may not
// read the full dataset.
class CachingTaskRunner : public TaskRunner {
 public:
  explicit CachingTaskRunner(std::unique_ptr<TaskIterator> iterator,
                             size_t max_cache_size_bytes);
  ~CachingTaskRunner() override;

  // Gets the next element from the cross-trainer cache, blocking if the data is
  // not ready.
  // REQUIRES: !req.trainer_id().empty()
  Status GetNext(const GetElementRequest& req,
                 GetElementResult& result) override;

  // Cancel the task runner. After cancelling, all the `GetNext` calls will
  // return a Cancelled status.
  void Cancel() override;

  // Returns the dataset model for performance analysis.
  std::shared_ptr<model::Model> model() const override;

 private:
  // The `GetElementResultSequence` generates a sequence of elements from the
  // `FirstComeFirstServedTaskRunner`. It is used for the `CrossTrainerCache` to
  // generate cached elements.
  class GetElementResultSequence : public CachableSequence<GetElementResult> {
   public:
    explicit GetElementResultSequence(
        FirstComeFirstServedTaskRunner& fcfs_task_runner);
    StatusOr<GetElementResult> GetNext() override;
    size_t GetElementSizeBytes(const GetElementResult& element) const override;

   private:
    FirstComeFirstServedTaskRunner& fcfs_task_runner_;
  };

  FirstComeFirstServedTaskRunner fcfs_task_runner_;
  CrossTrainerCache<GetElementResult> cache_;

  CachingTaskRunner(const CachingTaskRunner&) = delete;
  void operator=(const CachingTaskRunner&) = delete;
};

// An element produced by a task.
struct Element {
  explicit Element(std::vector<Tensor>&& components, int64_t index)
      : components(components), index(index) {}
  // The components of the element.
  std::vector<Tensor> components;
  // The element's index within the task, e.g. 0 for the first element produced
  // by the task, 1 for the second element, etc.
  int64_t index;
};

// Thread for prefetching a round worth of elements.
class PrefetchThread {
 public:
  explicit PrefetchThread(std::unique_ptr<TaskIterator> iterator,
                          int64_t round_size);
  ~PrefetchThread();
  // Runs the prefetch thread. It runs until an error is encountered or the
  // destructor is called.
  void Run();
  // Fills `out` with a round of data. Waits for up to `wait_us` microseconds
  // before giving up and returning with `out` empty. A negative `wait_us`
  // signals to wait indefinitely.
  Status FillBuffer(int64_t wait_us,
                    std::vector<std::unique_ptr<Element>>& out);
  // Returns the status for any failures encountered by the prefetch thread.
  Status GetStatus();
  // Returns the dataset model for performance analysis.
  std::shared_ptr<model::Model> model() const;

 private:
  const std::unique_ptr<TaskIterator> iterator_;
  const int64_t round_size_;
  mutex mu_;
  int64_t index_ TF_GUARDED_BY(mu_) = 0;
  // Buffered results for the next round.
  std::vector<std::unique_ptr<Element>> buffer_ TF_GUARDED_BY(mu_);
  // The status if the prefetch thread fails.
  Status status_ TF_GUARDED_BY(mu_) = absl::OkStatus();
  // Condition variable notified when elements are added to or removed from
  // `buffer_`, or when `status_` is changed.
  condition_variable cv_;
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
  // Thread which constantly tries to fill `buffer_` up with
  // `num_consumers` elements.
  std::unique_ptr<Thread> thread_;
};

// A task runner which enforces round-robin order for consuming a task's
// elements. `RoundRobinTaskRunner` provides elements in a series of "rounds".
// In each successive round, the runner waits to receive requests from all
// consumers. These requests are blocked until all requests arrive. Once all
// requests arrive, the runner hands out elements to consumers in order of their
// consumer indices.
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
                       int64_t num_consumers, string worker_address);

  Status GetNext(const GetElementRequest& req,
                 GetElementResult& result) override;
  void Cancel() override;
  std::shared_ptr<model::Model> model() const override;

 private:
  // Prepares a full round of data. `wait_us` indicates how long to wait before
  // skipping if a full round of data is not yet ready.
  Status PrepareFullRound(int64_t wait_us) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Prepares a partial round to get consumers back in sync.
  Status PreparePartialRound() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status ValidateRequest(const GetElementRequest& req);
  // Prepares data for the next round, blocking until the round is ready to
  // start.
  Status PrepareRound(const GetElementRequest& req);
  const int64_t num_consumers_;
  const string worker_address_;
  mutex mu_;
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
  // Condition variable notified whenever we start a new round of round-robin.
  condition_variable new_round_cv_;
  // Outstanding requests, indexed by round number and then consumer index.
  absl::flat_hash_map<int64_t,
                      absl::flat_hash_map<int64_t, const GetElementRequest*>>
      requests_ TF_GUARDED_BY(mu_);
  // Index of the first round we plan to serve. At startup, this is the minimum
  // of all requested element indices.
  int64_t first_round_ TF_GUARDED_BY(mu_) = kint64max;
  int64_t current_round_ TF_GUARDED_BY(mu_) = -1;
  bool round_skipped_ TF_GUARDED_BY(mu_) = false;
  // Buffered results for the current round.
  std::vector<std::unique_ptr<Element>> buffer_ TF_GUARDED_BY(mu_);
  // Thread which constantly tries to prepare `num_consumers` elements for the
  // next round.
  PrefetchThread prefetch_thread_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_TASK_RUNNER_H_
