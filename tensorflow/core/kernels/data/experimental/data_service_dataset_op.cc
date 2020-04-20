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

#include "grpcpp/create_channel.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/security/credentials.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/service/compression_utils.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/master.grpc.pb.h"
#include "tensorflow/core/data/service/master.pb.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
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
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const DataServiceDatasetOp::kDatasetType;
/* static */ constexpr const char* const DataServiceDatasetOp::kAddress;
/* static */ constexpr const char* const DataServiceDatasetOp::kProtocol;
/* static */ constexpr const char* const
    DataServiceDatasetOp::kMaxOutstandingRequests;
/* static */ constexpr const char* const DataServiceDatasetOp::kOutputTypes;
/* static */ constexpr const char* const DataServiceDatasetOp::kOutputShapes;

// Once we've spent `kRetryTimeoutMicros` in `GetNextInternal`, we will wait for
// the current attempt to complete and perform no more retries.
const int64 kRetryTimeoutMicros = 1000LL * 1000 * 60 * 60;  // 60 minutes.

// How often to refresh the task list.
const int64 kRefreshTasksIntervalMicros = 1000LL * 1000 * 60;  // 60 seconds.

// Dataset for reading data from the tf.data service non-deterministically.
//
// This dataset interleaves dataset elements produced by multiple tf.data
// workers. We periodically query the tf.data master to determine which workers
// to read from (in case workers are added or removed).
class DataServiceDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const std::string& address,
          const std::string& protocol, const int64 max_outstanding_requests,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        address_(address),
        protocol_(protocol),
        max_outstanding_requests_(max_outstanding_requests),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
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
    Node* address;
    TF_RETURN_IF_ERROR(b->AddScalar(address_, &address));

    Node* protocol;
    TF_RETURN_IF_ERROR(b->AddScalar(protocol_, &protocol));

    Node* max_outstanding_requests;
    TF_RETURN_IF_ERROR(
        b->AddScalar(max_outstanding_requests_, &max_outstanding_requests));

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {address, protocol, max_outstanding_requests}, {}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    ~Iterator() override {
      mutex_lock l(mu_);
      cancelled_ = true;
      cv_.notify_all();
      // Thread destructors will block until the threads finish, no need to wait
      // here.
    }

    Status Initialize(IteratorContext* ctx) override {
      VLOG(3) << "Connecting to " << dataset()->address_
              << " in data service dataset op";
      if (ctx->job_token().is_empty()) {
        return errors::FailedPrecondition(
            "Expected a job token, but none found. To iterate over a dataset "
            "containing a `distribute` transformation, call `create_job`, "
            "which will return a job token that you should then use to iterate "
            "over the dataset via `create_iterator(dataset, job_token).`");
      }
      job_id_ = ctx->job_token().job_id();
      TF_RETURN_IF_ERROR(CredentialsFactory::CreateClientCredentials(
          dataset()->protocol_, &credentials_));
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      VLOG(3) << "Calling GetNext in data service dataset op";
      mutex_lock l(mu_);
      if (!task_thread_manager_ && !cancelled_) {
        task_thread_manager_ = ctx->StartThread(
            "task-thread-manager", [this, ctx]() { TaskThreadManager(ctx); });
      }

      // tasks_.empty() indicates that we haven't yet received tasks from the
      // master, so we should wait.
      while (results_.empty() &&
             (tasks_.empty() || num_unfinished_tasks_ > 0) && !cancelled_) {
        cv_.wait(l);
      }
      if (cancelled_) {
        return errors::Cancelled("Data service iterator was cancelled");
      }
      if (results_.empty()) {
        *end_of_sequence = true;
        return Status::OK();
      }
      DCHECK(!results_.empty());
      out_tensors->swap(results_.front());
      results_.pop();
      cv_.notify_all();

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
    typedef struct TaskThread {
      int64 task_id;
      // Cached address of the worker for task `task_id`.
      std::string address;
      std::unique_ptr<WorkerService::Stub> worker_stub;
      std::unique_ptr<Thread> thread;
      bool end_of_sequence = false;
    } TaskThread;

    // Periodically refresh the task list.
    // Maintain one thread fetching elements for each task.
    // TODO(aaudibert): Instead of polling, have master send updates when
    // the list of tasks changes.
    void TaskThreadManager(IteratorContext* ctx) {
      VLOG(3) << "Starting task handler manager";
      auto channel = ::grpc::CreateChannel(dataset()->address_, credentials_);
      std::unique_ptr<MasterService::Stub> master_stub =
          MasterService::NewStub(channel);

      uint64 next_check = Env::Default()->NowMicros();
      while (true) {
        {
          mutex_lock l(mu_);
          // All units are microseconds.
          while (!cancelled_ && Env::Default()->NowMicros() < next_check) {
            int64 remaining_time = next_check - Env::Default()->NowMicros();
            VLOG(3) << "Task manager waiting for " << remaining_time << "us";
            cv_.wait_for(l, std::chrono::microseconds(remaining_time));
          }
          if (cancelled_) {
            return;
          }
        }
        UpdateTaskThreads(master_stub.get(), ctx);
        next_check = Env::Default()->NowMicros() + kRefreshTasksIntervalMicros;
      }
    }

    void UpdateTaskThreads(MasterService::Stub* master_stub,
                           IteratorContext* ctx) LOCKS_EXCLUDED(mu_) {
      VLOG(3) << "Updating task handler threads";
      GetTasksResponse resp;
      GetTasksRequest req;
      req.set_job_id(job_id_);
      grpc::ClientContext client_ctx;
      grpc::Status s = master_stub->GetTasks(&client_ctx, req, &resp);
      if (!s.ok()) {
        LOG(INFO) << "Failed to get task info for job id " << job_id_ << ": "
                  << s.error_message() << "(" << s.error_code() << ")";
        return;
      }
      absl::flat_hash_set<int64> task_ids;
      mutex_lock l(mu_);
      for (auto& task : resp.task_info()) {
        task_ids.insert(task.id());
        if (task_threads_.contains(task.id())) {
          continue;
        }
        tasks_[task.id()] = task;
        task_threads_[task.id()] = absl::make_unique<TaskThread>();
        TaskThread* task_handler = task_threads_[task.id()].get();
        task_handler->task_id = task.id();
        num_unfinished_tasks_++;
        task_handler->thread = ctx->StartThread(
            "tf-data-service-task_handler",
            [this, task_handler]() { RunTaskThread(task_handler); });
      }
      // Mark deleted tasks and clean up finished task threads.
      for (auto it = task_threads_.begin(); it != task_threads_.end();) {
        TaskThread* task_thread = it->second.get();
        if (task_thread->end_of_sequence) {
          task_threads_.erase(it++);
          continue;
        }
        if (!task_ids.contains(task_thread->task_id)) {
          task_thread->end_of_sequence = true;
        }
        ++it;
      }
      if (dataset()->max_outstanding_requests_ == model::kAutotune) {
        // Adjust max_outstanding_requests to account for newly added tasks.
        max_outstanding_requests_ = task_threads_.size();
      }
    }

    void RunTaskThread(TaskThread* task_handler) {
      auto cleanup = gtl::MakeCleanup([this]() {
        mutex_lock l(mu_);
        outstanding_requests_--;
        num_unfinished_tasks_--;
        cv_.notify_all();
      });
      {
        mutex_lock l(mu_);
        outstanding_requests_++;
        task_handler->address = tasks_[task_handler->task_id].worker_address();
      }
      VLOG(3) << "Starting task handler thread for task "
              << task_handler->task_id << " with worker address "
              << task_handler->address;
      while (true) {
        if (!task_handler->worker_stub) {
          Status s = CreateWorkerStub(task_handler->address,
                                      &task_handler->worker_stub);
          if (!s.ok()) {
            LOG(WARNING) << "Failed to create a worker stub for "
                         << task_handler->address << ": " << s;
          }
        }
        {
          mutex_lock l(mu_);
          if (task_handler->end_of_sequence) {
            return;
          }
          outstanding_requests_--;
          while (!cancelled_ && results_.size() + outstanding_requests_ >=
                                    max_outstanding_requests_) {
            VLOG(3) << "Task thread for task " << task_handler->task_id
                    << " waiting. results_.size()=" << results_.size()
                    << " outstanding_requests_=" << outstanding_requests_;
            cv_.wait(l);
          }
          outstanding_requests_++;
          if (cancelled_) {
            return;
          }
        }
        // TODO(aaudibert): add backoff and max retries.
        int64 deadline_micros =
            Env::Default()->NowMicros() + kRetryTimeoutMicros;
        Status s = FetchElement(task_handler, deadline_micros);
        if (!s.ok()) {
          LOG(WARNING) << "Failed to fetch element from worker at "
                       << task_handler->address << ": " << s;
        }
      }
    }

    Status FetchElement(TaskThread* task_handler, int64 deadline_micros) {
      VLOG(3) << "Fetchng an element for task id " << task_handler->task_id;
      GetElementResponse resp;
      TF_RETURN_IF_ERROR(
          GetElementWithDeadline(task_handler, &resp, deadline_micros));
      std::vector<Tensor> element;
      if (!resp.end_of_sequence()) {
        TF_RETURN_IF_ERROR(
            service_util::Uncompress(resp.compressed_element(), &element));
      }
      mutex_lock l(mu_);
      if (resp.end_of_sequence()) {
        task_handler->end_of_sequence = true;
        return Status::OK();
      }
      results_.push(std::move(element));
      cv_.notify_all();
      VLOG(3) << "Fetched an element for task id " << task_handler->task_id;
      return Status::OK();
    }

    Status CreateWorkerStub(const std::string& worker_address,
                            std::unique_ptr<WorkerService::Stub>* stub) {
      ::grpc::ChannelArguments args;
      args.SetMaxReceiveMessageSize(-1);
      std::shared_ptr<::grpc::ChannelCredentials> credentials;
      TF_RETURN_IF_ERROR(CredentialsFactory::CreateClientCredentials(
          dataset()->protocol_, &credentials));
      auto channel =
          ::grpc::CreateCustomChannel(worker_address, credentials, args);
      *stub = WorkerService::NewStub(channel);
      return Status::OK();
    }

    Status GetElementWithDeadline(TaskThread* task_handler,
                                  GetElementResponse* resp,
                                  int64 deadline_micros) {
      return RetryWithDeadline(
          [task_handler, resp] {
            GetElementRequest req;
            req.set_task_id(task_handler->task_id);
            grpc::ClientContext client_ctx;
            grpc::Status s =
                task_handler->worker_stub->GetElement(&client_ctx, req, resp);
            if (s.ok()) {
              return Status::OK();
            }
            return grpc_util::WrapError("Failed to fetch an element", s);
          },
          deadline_micros);
    }

    static bool ShouldRetryError(error::Code error_code) {
      // Retry all errors that could indicate preemption.
      return error_code == error::Code::UNAVAILABLE ||
             error_code == error::Code::CANCELLED ||
             error_code == error::Code::ABORTED;
    }

    static Status RetryWithDeadline(const std::function<Status()>& call,
                                    int64 deadline_micros) {
      Status s;
      for (int num_retries = 0;; ++num_retries) {
        s = call();
        if (s.ok() || !ShouldRetryError(s.code())) {
          return s;
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
    }

    mutex mu_;
    // TODO(aaudibert): split this into a couple cvs for different conditions
    // so that we can use notify_one and avoid unnecessary wakeups.
    condition_variable cv_ TF_GUARDED_BY(mu_);
    bool cancelled_ TF_GUARDED_BY(mu_) = false;

    int64 outstanding_requests_ TF_GUARDED_BY(mu_) = 0;
    // max_outstanding_requests controls how many elements may be held in memory
    // at the same time. This count includes both in-progress requests for
    // elements as well as completed requests which haven't yet been produced.
    int64 max_outstanding_requests_ TF_GUARDED_BY(mu_);
    std::queue<std::vector<Tensor>> results_ TF_GUARDED_BY(mu_);

    // Set once in Initialize().
    int64 job_id_;
    std::shared_ptr<::grpc::ChannelCredentials> credentials_;
    int64 num_unfinished_tasks_ TF_GUARDED_BY(mu_) = 0;
    // Map from task id to task info.
    absl::flat_hash_map<int64, TaskInfo> tasks_ TF_GUARDED_BY(mu_);

    // Must come second to last so that task threads are joined before
    // destroying other fields.
    absl::flat_hash_map<int64, std::unique_ptr<TaskThread>> task_threads_
        TF_GUARDED_BY(mu_);
    // Must be ordered last so that the thread is joined before destroying other
    // fields.
    std::unique_ptr<Thread> task_thread_manager_ GUARDED_BY(mu_);
  };

  const tstring address_;
  const tstring protocol_;
  const int64 max_outstanding_requests_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

DataServiceDatasetOp::DataServiceDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void DataServiceDatasetOp::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
  tstring address;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kAddress, &address));
  OP_REQUIRES(ctx, !address.empty(),
              errors::InvalidArgument(kAddress, " must be non-empty."));

  tstring protocol;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kProtocol, &protocol));
  OP_REQUIRES(ctx, !protocol.empty(),
              errors::InvalidArgument(kProtocol, " must be non-empty."));

  int64 max_outstanding_requests;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kMaxOutstandingRequests,
                                          &max_outstanding_requests));
  OP_REQUIRES(
      ctx,
      max_outstanding_requests == model::kAutotune ||
          max_outstanding_requests > 0,
      errors::InvalidArgument(kMaxOutstandingRequests, " must be positive or ",
                              model::kAutotune));

  *output = new Dataset(ctx, address, protocol, max_outstanding_requests,
                        output_types_, output_shapes_);
}

REGISTER_KERNEL_BUILDER(Name("DataServiceDataset").Device(DEVICE_CPU),
                        DataServiceDatasetOp);

}  // namespace data
}  // namespace tensorflow
