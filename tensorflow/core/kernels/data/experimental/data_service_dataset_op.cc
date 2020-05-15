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
#include "tensorflow/core/kernels/data/experimental/data_service_dataset_op.h"

#include <map>
#include <memory>
#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/compression_utils.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/serialization_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const DataServiceDatasetOp::kDatasetType;
/* static */ constexpr const char* const DataServiceDatasetOp::kDatasetId;
/* static */ constexpr const char* const DataServiceDatasetOp::kProcessingMode;
/* static */ constexpr const char* const DataServiceDatasetOp::kAddress;
/* static */ constexpr const char* const DataServiceDatasetOp::kProtocol;
/* static */ constexpr const char* const DataServiceDatasetOp::kJobName;
/* static */ constexpr const char* const
    DataServiceDatasetOp::kMaxOutstandingRequests;
/* static */ constexpr const char* const
    DataServiceDatasetOp::kIterationCounter;
/* static */ constexpr const char* const DataServiceDatasetOp::kOutputTypes;
/* static */ constexpr const char* const DataServiceDatasetOp::kOutputShapes;

namespace {
// Once we've spent `kRetryTimeoutMicros` in `GetNextInternal`, we will wait for
// the current attempt to complete and perform no more retries.
const int64 kRetryTimeoutMicros = 1000LL * 1000 * 60 * 60;  // 60 minutes.

// Default interval between task list refreshes.
const int64 kDefaultTaskRefreshIntervalMs = 1000;  // 1 second.

}  // namespace

// Dataset for reading data from the tf.data service non-deterministically.
//
// This dataset interleaves dataset elements produced by multiple tf.data
// workers. We periodically query the tf.data master to determine which workers
// to read from (in case workers are added or removed).
class DataServiceDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64 dataset_id,
          ProcessingMode processing_mode, const std::string& address,
          const std::string& protocol, const std::string& job_name,
          int64 max_outstanding_requests, int64 task_refresh_interval_ms,
          IterationCounter* iteration_counter, bool owns_resource,
          ResourceHandle iteration_counter_handle,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        dataset_id_(dataset_id),
        processing_mode_(processing_mode),
        address_(address),
        protocol_(protocol),
        job_name_(job_name),
        max_outstanding_requests_(max_outstanding_requests),
        task_refresh_interval_ms_(task_refresh_interval_ms),
        iteration_counter_(iteration_counter),
        owns_resource_(owns_resource),
        iteration_counter_handle_(iteration_counter_handle),
        resource_mgr_(ctx->resource_manager()),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  ~Dataset() override {
    iteration_counter_->Unref();
    if (owns_resource_) {
      Status s = resource_mgr_->Delete<IterationCounter>(
          iteration_counter_handle_.container(),
          iteration_counter_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete iteration counter resource: " << s;
      }
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this,
                         name_utils::IteratorPrefix(kDatasetType, prefix)},
        iteration_counter_->GetAndIncrement());
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status CheckExternalState() const override {
    return Status(
        error::FAILED_PRECONDITION,
        strings::StrCat(DebugString(), " does not yet support serialization."));
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* dataset_id;
    TF_RETURN_IF_ERROR(b->AddScalar(dataset_id_, &dataset_id));

    Node* processing_mode;
    tstring processing_mode_str = ProcessingModeToString(processing_mode_);
    TF_RETURN_IF_ERROR(b->AddScalar(processing_mode_str, &processing_mode));

    Node* address;
    TF_RETURN_IF_ERROR(b->AddScalar(address_, &address));

    Node* protocol;
    TF_RETURN_IF_ERROR(b->AddScalar(protocol_, &protocol));

    Node* job_name;
    TF_RETURN_IF_ERROR(b->AddScalar(job_name_, &job_name));

    Node* max_outstanding_requests;
    TF_RETURN_IF_ERROR(
        b->AddScalar(max_outstanding_requests_, &max_outstanding_requests));

    Node* iteration_counter_handle = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = iteration_counter_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &iteration_counter_handle));

    AttrValue task_refresh_interval_hint_ms;
    b->BuildAttrValue(task_refresh_interval_ms_,
                      &task_refresh_interval_hint_ms);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this,
                      {dataset_id, processing_mode, address, protocol, job_name,
                       max_outstanding_requests, iteration_counter_handle},
                      {std::make_pair(kTaskRefreshIntervalHintMs,
                                      task_refresh_interval_hint_ms)},
                      output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params, int64 iterator_index)
        : DatasetIterator<Dataset>(params),
          iterator_index_(iterator_index),
          max_outstanding_requests_(params.dataset->max_outstanding_requests_) {
    }

    ~Iterator() override {
      mutex_lock l(mu_);
      VLOG(1) << "Destroying data service dataset iterator for job id "
              << job_id_;
      cancelled_ = true;
      worker_thread_cv_.notify_all();
      manager_thread_cv_.notify_all();
      get_next_cv_.notify_all();
      // Thread destructors will block until the threads finish, no need to wait
      // here.
    }

    Status Initialize(IteratorContext* ctx) override {
      VLOG(3) << "Connecting to " << dataset()->address_
              << " in data service dataset op";
      DataServiceMasterClient master(dataset()->address_, dataset()->protocol_);
      if (dataset()->job_name_.empty()) {
        TF_RETURN_IF_ERROR(master.CreateJob(
            dataset()->dataset_id_, dataset()->processing_mode_, &job_id_));
      } else {
        TF_RETURN_IF_ERROR(master.GetOrCreateJob(
            dataset()->dataset_id_, dataset()->processing_mode_,
            dataset()->job_name_, iterator_index_, &job_id_));
      }
      VLOG(1) << "Created data service job with id " << job_id_;
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      VLOG(3) << "Calling GetNext in data service dataset op";
      mutex_lock l(mu_);
      if (!task_thread_manager_ && !cancelled_) {
        task_thread_manager_ =
            ctx->StartThread("task-thread-manager", [this, ctx]() {
              TaskThreadManager(absl::make_unique<IteratorContext>(*ctx));
            });
      }

      while (results_.empty() && !job_finished_ && !cancelled_ &&
             status_.ok()) {
        get_next_cv_.wait(l);
      }
      if (cancelled_) {
        return errors::Cancelled("Data service iterator was cancelled");
      }
      if (!status_.ok()) {
        return status_;
      }
      if (results_.empty()) {
        *end_of_sequence = true;
        return Status::OK();
      }
      DCHECK(!results_.empty());
      *end_of_sequence = false;
      out_tensors->swap(results_.front());
      results_.pop();
      worker_thread_cv_.notify_one();

      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return errors::Unimplemented("SaveInternal is not yet supported");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return errors::Unimplemented("RestoreInternal is not yet supported");
    }

   private:
    struct Task {
      Task(int64 task_id, const std::string& address,
           std::unique_ptr<DataServiceWorkerClient> worker)
          : task_id(task_id), address(address), worker(std::move(worker)) {}

      const int64 task_id;
      // Address of the tf.data service worker for task `task_id`.
      const std::string address;
      // Client for fetching task elements from the tf.data service worker.
      const std::unique_ptr<DataServiceWorkerClient> worker;
      // Indicates whether a worker thread is currently processing the task.
      bool in_use TF_GUARDED_BY(&Iterator::mu_) = false;
      // Indicates whether the worker has returned end_of_sequence for the task.
      bool end_of_sequence TF_GUARDED_BY(&Iterator::mu_) = false;
    };

    // Periodically refresh the task list.
    // Maintain one thread fetching elements for each task.
    // TODO(aaudibert): Instead of polling, have master send updates when
    // the list of tasks changes.
    void TaskThreadManager(std::unique_ptr<IteratorContext> ctx) {
      VLOG(3) << "Starting task thread manager";
      DataServiceMasterClient master(dataset()->address_, dataset()->protocol_);
      uint64 next_check = Env::Default()->NowMicros();
      while (true) {
        {
          mutex_lock l(mu_);
          // All units are microseconds.
          while (!cancelled_ && Env::Default()->NowMicros() < next_check) {
            int64 remaining_time = next_check - Env::Default()->NowMicros();
            VLOG(3) << "Task thread manager waiting for " << remaining_time
                    << "us";
            manager_thread_cv_.wait_for(
                l, std::chrono::microseconds(remaining_time));
          }
          if (cancelled_) {
            VLOG(3) << "Task thread manager finished";
            return;
          }
        }
        UpdateTasks(&master);
        UpdateWorkerThreads(ctx.get());
        next_check = Env::Default()->NowMicros() +
                     dataset()->task_refresh_interval_ms_ * 1000;
      }
    }

    void UpdateTasks(DataServiceMasterClient* master) LOCKS_EXCLUDED(mu_) {
      VLOG(3) << "Updating tasks";
      std::vector<TaskInfo> tasks;
      bool job_finished;
      Status s = master->GetTasks(job_id_, &tasks, &job_finished);
      if (!s.ok()) {
        LOG(WARNING) << "Failed to get task info for job id " << job_id_ << ": "
                     << s;
        return;
      }
      absl::flat_hash_map<int64, TaskInfo> task_id_to_task;
      for (auto& task : tasks) {
        task_id_to_task[task.id()] = task;
      }
      mutex_lock l(mu_);
      job_finished_ = job_finished;
      if (job_finished) {
        get_next_cv_.notify_all();
        return;
      }
      for (int i = 0; i < tasks_.size(); ++i) {
        std::shared_ptr<Task> task = tasks_[i];
        if (task_id_to_task.contains(task->task_id)) {
          // Remove already-known tasks from `task_id_to_task`, so that at the
          // end of the loop, only new tasks remain.
          task_id_to_task.erase(task->task_id);
        } else {
          // Task has been removed.
          if (task->end_of_sequence) {
            finished_tasks_--;
          }
          tasks_[i] = tasks_[tasks_.size() - 1];
          tasks_.pop_back();
        }
      }
      for (auto& new_task_entry : task_id_to_task) {
        TaskInfo& task_info = new_task_entry.second;
        std::unique_ptr<DataServiceWorkerClient> worker;
        Status s = CreateDataServiceWorkerClient(task_info.worker_address(),
                                                 dataset()->protocol_, &worker);
        if (!s.ok()) {
          status_ = s;
          get_next_cv_.notify_all();
          continue;
        }
        tasks_.push_back(std::make_shared<Task>(
            task_info.id(), task_info.worker_address(), std::move(worker)));
      }
      if (dataset()->max_outstanding_requests_ == model::kAutotune) {
        // Adjust max_outstanding_requests to account for newly added tasks.
        max_outstanding_requests_ = tasks_.size();
      }
    }

    void UpdateWorkerThreads(IteratorContext* ctx) LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      while (num_running_worker_threads_ < max_outstanding_requests_) {
        num_running_worker_threads_++;
        outstanding_requests_++;
        auto done = [this]() {
          mutex_lock l(mu_);
          num_running_worker_threads_--;
          outstanding_requests_--;
          VLOG(3) << "Exiting worker thread";
        };
        worker_threads_.push_back(ctx->StartThread(
            "tf-data-service-task_thread", [this, done = std::move(done)]() {
              RunWorkerThread(std::move(done));
            }));
      }
    }

    void RunWorkerThread(std::function<void()> done) {
      auto cleanup = gtl::MakeCleanup([done = std::move(done)]() { done(); });
      VLOG(3) << "Starting worker thread";
      std::shared_ptr<Task> task_to_process;
      while (true) {
        {
          mutex_lock l(mu_);
          if (task_to_process) {
            task_to_process->in_use = false;
            task_to_process = nullptr;
            worker_thread_cv_.notify_one();
          }
          outstanding_requests_--;
          while (!cancelled_ && !(SpaceInBuffer() && TaskAvailable())) {
            if (VLOG_IS_ON(3)) {
              VLOG(3) << "Sleeping with results_.size=" << results_.size()
                      << ", outstanding_requests_=" << outstanding_requests_
                      << ", max_oustanding_requests="
                      << max_outstanding_requests_
                      << " finished_tasks=" << finished_tasks_
                      << " tasks_.size()=" << tasks_.size();
            }
            worker_thread_cv_.wait(l);
          }
          if (cancelled_) {
            return;
          }
          outstanding_requests_++;
          // Search for a task to update.
          int num_tasks = tasks_.size();
          for (int i = 0; i < num_tasks; ++i) {
            int index = (next_task_index_ + i) % num_tasks;
            std::shared_ptr<Task>& task = tasks_[index];
            if (!task->in_use && !task->end_of_sequence) {
              task->in_use = true;
              task_to_process = task;
              next_task_index_ = (index + 1) % num_tasks;
              break;
            }
          }
          DCHECK(task_to_process != nullptr);
          VLOG(3) << "Processing task " << task_to_process->task_id;
        }
        int64 deadline_micros =
            Env::Default()->NowMicros() + kRetryTimeoutMicros;
        Status s = GetElement(task_to_process.get(), deadline_micros);
        if (!s.ok()) {
          mutex_lock l(mu_);
          status_ = s;
          get_next_cv_.notify_all();
          return;
        }
      }
    }

    // Gets an element from a task and adds the element to `results_`.
    //
    // If the task reaches end_of_sequence or is cancelled (e.g. due to a
    // worker dying), GetElement returns Status::OK() without adding to
    // `results_`.
    Status GetElement(Task* task, int64 deadline_micros)
        TF_LOCKS_EXCLUDED(mu_) {
      VLOG(3) << "Getting an element for task id " << task->task_id;
      tensorflow::profiler::TraceMe activity(
          "GetElement", tensorflow::profiler::TraceMeLevel::kInfo);
      CompressedElement compressed;
      bool end_of_sequence;
      for (int num_retries = 0;; ++num_retries) {
        Status s = task->worker->GetElement(task->task_id, &compressed,
                                            &end_of_sequence);
        if (s.ok()) {
          break;
        }
        if (errors::IsNotFound(s)) {
          // This indicates that the worker was restarted. The restarted worker
          // will get a new task, and the old task is lost.
          mutex_lock l(mu_);
          finished_tasks_++;
          task->end_of_sequence = true;
          return Status::OK();
        }
        // Retry all errors that could indicate preemption.
        if (!errors::IsUnavailable(s) && !errors::IsCancelled(s) &&
            !errors::IsAborted(s)) {
          return s;
        }
        {
          mutex_lock l(mu_);
          // If `UpdateTaskThreads` finds that the task has been cancelled, it
          // will set end_of_sequence to `true`.
          if (task->end_of_sequence || cancelled_) {
            return Status::OK();
          }
        }
        const int64 now_micros = EnvTime::NowMicros();
        if (now_micros > deadline_micros) {
          return s;
        }
        const int64 deadline_with_backoff_micros =
            now_micros + ::tensorflow::ComputeBackoffMicroseconds(num_retries);
        // Wait for a short period of time before retrying the RPC. If our
        // backoff would put us past the RPC deadline, we truncate it to ensure
        // our RPC starts before the deadline.
        const auto backoff_until =
            (deadline_micros > deadline_with_backoff_micros)
                ? deadline_with_backoff_micros
                : deadline_micros;
        Env::Default()->SleepForMicroseconds(backoff_until - now_micros);
      }

      std::vector<Tensor> element;
      if (!end_of_sequence) {
        TF_RETURN_IF_ERROR(service_util::Uncompress(compressed, &element));
      }
      mutex_lock l(mu_);
      if (end_of_sequence) {
        task->end_of_sequence = true;
        finished_tasks_++;
        return Status::OK();
      }
      results_.push(std::move(element));
      get_next_cv_.notify_all();
      VLOG(3) << "Got an element for task id " << task->task_id;
      return Status::OK();
    }

    bool SpaceInBuffer() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return results_.size() + outstanding_requests_ <
             max_outstanding_requests_;
    }

    bool TaskAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return finished_tasks_ + outstanding_requests_ < tasks_.size();
    }

    const int64 iterator_index_;

    mutex mu_;
    condition_variable get_next_cv_ TF_GUARDED_BY(mu_);
    condition_variable worker_thread_cv_ TF_GUARDED_BY(mu_);
    condition_variable manager_thread_cv_ TF_GUARDED_BY(mu_);
    bool cancelled_ TF_GUARDED_BY(mu_) = false;

    int64 outstanding_requests_ TF_GUARDED_BY(mu_) = 0;
    // max_outstanding_requests controls how many elements may be held in memory
    // at the same time. This count includes both in-progress requests for
    // elements as well as completed requests which haven't yet been produced.
    int64 max_outstanding_requests_ TF_GUARDED_BY(mu_);

    // The number of threads in `worker_threads_` which are still running.
    int64 num_running_worker_threads_ TF_GUARDED_BY(mu_) = 0;

    // The index of the next task in `tasks_` to read from.
    int64 next_task_index_ TF_GUARDED_BY(mu_) = 0;

    // The number tasks in the `tasks_` list that have reached end_of_sequence.
    int64 finished_tasks_ TF_GUARDED_BY(mu_) = 0;

    // List of tasks to read from.
    std::vector<std::shared_ptr<Task>> tasks_ TF_GUARDED_BY(mu_);

    // A status to be returned from the next call to `GetNext`. This is set by
    // asynchronous threads when they encounter errors.
    Status status_ TF_GUARDED_BY(mu_) = Status::OK();
    std::queue<std::vector<Tensor>> results_ TF_GUARDED_BY(mu_);

    // Set once in Initialize().
    int64 job_id_;

    bool job_finished_ = false;
    // Must be ordered second to last so that worker threads are joined before
    // destroying other fields.
    std::vector<std::unique_ptr<Thread>> worker_threads_ TF_GUARDED_BY(mu_);
    // Must be ordered last so that the thread is joined before destroying other
    // fields.
    std::unique_ptr<Thread> task_thread_manager_ GUARDED_BY(mu_);
  };

  const int64 dataset_id_;
  const ProcessingMode processing_mode_;
  const tstring address_;
  const tstring protocol_;
  const tstring job_name_;
  const int64 max_outstanding_requests_;
  const int64 task_refresh_interval_ms_;
  IterationCounter* const iteration_counter_;  // Owned
  const bool owns_resource_;
  const ResourceHandle iteration_counter_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

DataServiceDatasetOp::DataServiceDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kTaskRefreshIntervalHintMs,
                                   &task_refresh_interval_hint_ms_));
  if (task_refresh_interval_hint_ms_ == model::kAutotune) {
    task_refresh_interval_hint_ms_ = kDefaultTaskRefreshIntervalMs;
  }
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void DataServiceDatasetOp::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
  int64 dataset_id;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kDatasetId, &dataset_id));

  tstring processing_mode_str;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kProcessingMode, &processing_mode_str));
  ProcessingMode processing_mode;
  OP_REQUIRES_OK(ctx,
                 ParseProcessingMode(processing_mode_str, &processing_mode));

  tstring address;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kAddress, &address));
  OP_REQUIRES(ctx, !address.empty(),
              errors::InvalidArgument(kAddress, " must be non-empty."));

  tstring protocol;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kProtocol, &protocol));
  OP_REQUIRES(ctx, !protocol.empty(),
              errors::InvalidArgument(kProtocol, " must be non-empty."));

  tstring job_name;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kJobName, &job_name));

  int64 max_outstanding_requests;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kMaxOutstandingRequests,
                                          &max_outstanding_requests));

  ResourceHandle iteration_counter_handle;
  OP_REQUIRES_OK(
      ctx, HandleFromInput(ctx, kIterationCounter, &iteration_counter_handle));
  IterationCounter* iteration_counter = nullptr;
  Status s = ctx->resource_manager()->Lookup<IterationCounter>(
      iteration_counter_handle.container(), iteration_counter_handle.name(),
      &iteration_counter);
  bool owns_resource = false;
  if (errors::IsNotFound(s)) {
    owns_resource = true;
    static std::atomic<int64> resource_id_counter(0);
    const std::string& container = ctx->resource_manager()->default_container();
    std::string name =
        strings::StrCat(ctx->op_kernel().name(), "/", kIterationCounter, "_",
                        resource_id_counter.fetch_add(1));
    OP_REQUIRES_OK(ctx,
                   ctx->resource_manager()->LookupOrCreate<IterationCounter>(
                       container, name, &iteration_counter,
                       [](IterationCounter** counter) {
                         *counter = new IterationCounter();
                         return Status::OK();
                       }));
    iteration_counter_handle =
        MakeResourceHandle<IterationCounter>(ctx, container, name);
  } else {
    OP_REQUIRES_OK(ctx, s);
  }

  OP_REQUIRES(
      ctx,
      max_outstanding_requests == model::kAutotune ||
          max_outstanding_requests > 0,
      errors::InvalidArgument(kMaxOutstandingRequests, " must be positive or ",
                              model::kAutotune));

  *output =
      new Dataset(ctx, dataset_id, processing_mode, address, protocol, job_name,
                  max_outstanding_requests, task_refresh_interval_hint_ms_,
                  iteration_counter, owns_resource, iteration_counter_handle,
                  output_types_, output_shapes_);
}

REGISTER_KERNEL_BUILDER(Name("DataServiceDataset").Device(DEVICE_CPU),
                        DataServiceDatasetOp);
REGISTER_KERNEL_BUILDER(Name("DummyIterationCounter").Device(DEVICE_CPU),
                        DummyResourceOp<IterationCounter>);

}  // namespace data
}  // namespace tensorflow
