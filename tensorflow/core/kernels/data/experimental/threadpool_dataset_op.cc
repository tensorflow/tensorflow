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
#include "tensorflow/core/kernels/data/experimental/threadpool_dataset_op.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const
    MaxIntraOpParallelismDatasetOp::kDatasetType;
/* static */ constexpr const char* const
    MaxIntraOpParallelismDatasetOp::kDatasetOp;
/* static */ constexpr const char* const
    PrivateThreadPoolDatasetOp::kDatasetType;
/* static */ constexpr const char* const PrivateThreadPoolDatasetOp::kDatasetOp;

namespace {
// To prevent integer overflow issues when allocating threadpool memory for an
// unreasonable number of threads.
constexpr int kThreadLimit = 65536;

absl::Status ValidateNumThreads(int32_t num_threads) {
  if (num_threads < 0) {
    return errors::InvalidArgument("`num_threads` must be >= 0");
  }
  if (num_threads >= kThreadLimit) {
    return errors::InvalidArgument("`num_threads` must be < ", kThreadLimit);
  }
  return absl::OkStatus();
}
}  // namespace

class ThreadPoolResource : public ResourceBase {
 public:
  ThreadPoolResource(Env* env, const ThreadOptions& thread_options,
                     const string& name, int num_threads, bool low_latency_hint,
                     int max_intra_op_parallelism)
      : thread_pool_(env, thread_options, name, num_threads, low_latency_hint),
        max_intra_op_parallelism_(max_intra_op_parallelism) {}

  // Schedules fn() for execution in the pool of threads.
  void Schedule(std::function<void()> fn) {
    if (max_intra_op_parallelism_ < 0) {
      thread_pool_.Schedule(std::move(fn));
    } else {
      thread_pool_.Schedule(std::bind(
          [this](std::function<void()> bound_fn) {
            // TODO(mrry): Consider moving this thread-local configuration to
            // the threads themselves.
            ScopedPerThreadMaxParallelism scope(max_intra_op_parallelism_);
            bound_fn();
          },
          std::move(fn)));
    }
  }

  int32 NumThreads() { return thread_pool_.NumThreads(); }

  string DebugString() const override { return "ThreadPoolResource"; }

 private:
  thread::ThreadPool thread_pool_;
  const int max_intra_op_parallelism_;
};

// Creates a handle to a ThreadPool resource. Note that we don't use
// ResourceOpKernel here because the ThreadPoolResource constructor requires
// access to `OpKernelContext::env()`, which isn't provided by
// `ResourceOpKernel<T>::CreateResource()`.
class ThreadPoolHandleOp : public OpKernel {
 public:
  explicit ThreadPoolHandleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("display_name", &display_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_intra_op_parallelism",
                                     &max_intra_op_parallelism_));
    OP_REQUIRES_OK(ctx, ValidateNumThreads(num_threads_));

    // For consistency with Dataset, use MaxParallelism if 0 threads are
    // specified.
    if (num_threads_ == 0) {
      num_threads_ = port::MaxParallelism();
    }
  }

  // The resource is deleted from the resource manager only when it is private
  // to kernel. Ideally the resource should be deleted when it is no longer held
  // by anyone, but it would break backward compatibility.
  ~ThreadPoolHandleOp() override {
    if (cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->Delete<ThreadPoolResource>(cinfo_.container(), cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

  void Compute(OpKernelContext* ctx) override TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (!initialized_) {
      ResourceMgr* mgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, cinfo_.Init(mgr, def()));
      ThreadPoolResource* resource;
      OP_REQUIRES_OK(ctx, mgr->LookupOrCreate<ThreadPoolResource>(
                              cinfo_.container(), cinfo_.name(), &resource,
                              [this, ctx](ThreadPoolResource** ret)
                                  TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                                    *ret = new ThreadPoolResource(
                                        ctx->env(), {}, display_name_,
                                        num_threads_,
                                        /*low_latency_hint=*/false,
                                        max_intra_op_parallelism_);
                                    return absl::OkStatus();
                                  }));
      initialized_ = true;
    }
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            TypeIndex::Make<ThreadPoolResource>()));
  }

 private:
  mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
  bool initialized_ TF_GUARDED_BY(mu_) = false;
  string display_name_;
  int num_threads_;
  int max_intra_op_parallelism_;
};

class ThreadPoolDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ThreadPoolDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    core::RefCountPtr<ThreadPoolResource> threadpool_resource;
    ResourceHandle handle;
    OP_REQUIRES_OK(ctx, HandleFromInput(ctx, 1, &handle));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, handle, &threadpool_resource));
    *output = new Dataset(ctx, input, ctx->input(1), threadpool_resource.get());
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const Tensor& resource_handle, ThreadPoolResource* threadpool)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          resource_handle_(resource_handle),
          threadpool_(threadpool) {
      input_->Ref();
      threadpool_->Ref();
    }

    ~Dataset() override {
      input_->Unref();
      threadpool_->Unref();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::ThreadPool")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override {
      return "ThreadPoolDatasetOp::Dataset";
    }

    int64_t CardinalityInternal(CardinalityOptions options) const override {
      return input_->Cardinality(options);
    }

    absl::Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->push_back(input_);
      return absl::OkStatus();
    }

    absl::Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

   protected:
    absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                    DatasetGraphDefBuilder* b,
                                    Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* resource_handle_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddTensor(resource_handle_, &resource_handle_node));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_graph_node, resource_handle_node}, output));
      return absl::OkStatus();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      absl::Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(
            IteratorContext(CreateParams(ctx)), this, prefix(), &input_impl_);
      }

      absl::Status GetNextInternal(IteratorContext* ctx,
                                   std::vector<Tensor>* out_tensors,
                                   bool* end_of_sequence) override {
        return input_impl_->GetNext(IteratorContext(CreateParams(ctx)),
                                    out_tensors, end_of_sequence);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      absl::Status SaveInternal(SerializationContext* ctx,
                                IteratorStateWriter* writer) override {
        DCHECK(input_impl_ != nullptr);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        return absl::OkStatus();
      }

      absl::Status RestoreInternal(IteratorContext* ctx,
                                   IteratorStateReader* reader) override {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        return absl::OkStatus();
      }

     private:
      IteratorContext::Params CreateParams(IteratorContext* ctx) {
        ThreadPoolResource* pool = dataset()->threadpool_;
        IteratorContext::Params params(ctx);
        params.runner = [pool](std::function<void()> c) {
          pool->Schedule(std::move(c));
        };
        params.runner_threadpool_size = pool->NumThreads();
        return params;
      }

      std::unique_ptr<IteratorBase> input_impl_;
    };

    const DatasetBase* const input_;
    const Tensor resource_handle_;
    ThreadPoolResource* const threadpool_;
  };
};

class MaxIntraOpParallelismDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          int64_t max_intra_op_parallelism)
      : Dataset(DatasetContext(ctx), input, max_intra_op_parallelism) {}

  Dataset(DatasetContext&& ctx, const DatasetBase* input,
          int64_t max_intra_op_parallelism)
      : DatasetBase(std::move(ctx)),
        input_(input),
        max_intra_op_parallelism_(max_intra_op_parallelism),
        traceme_metadata_(
            {{"parallelism",
              strings::Printf("%lld", static_cast<long long>(
                                          max_intra_op_parallelism_))}}) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, strings::StrCat(prefix, "::MaxIntraOpParallelism")});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return "MaxIntraOpParallelismDatasetOp::Dataset";
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return input_->Cardinality(options);
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->clear();
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* max_intra_op_parallelism_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(max_intra_op_parallelism_,
                                    &max_intra_op_parallelism_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, max_intra_op_parallelism_node}, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    absl::Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      IteratorContext::Params params(ctx);
      auto max_parallelism = dataset()->max_intra_op_parallelism_;
      params.runner = RunnerWithMaxParallelism(*ctx->runner(), max_parallelism);
      return input_impl_->GetNext(IteratorContext{std::move(params)},
                                  out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      DCHECK(input_impl_ != nullptr);
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      return absl::OkStatus();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      return dataset()->traceme_metadata_;
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
  };

  const DatasetBase* const input_;
  const int64_t max_intra_op_parallelism_;
  const TraceMeMetadata traceme_metadata_;
};

/* static */
void MaxIntraOpParallelismDatasetOp::MakeDatasetFromOptions(
    OpKernelContext* ctx, DatasetBase* input, int32_t max_intra_op_parallelism,
    DatasetBase** output) {
  OP_REQUIRES(
      ctx, max_intra_op_parallelism >= 0,
      errors::InvalidArgument("`max_intra_op_parallelism` must be >= 0"));
  *output = new Dataset(DatasetContext(DatasetContext::Params(
                            {MaxIntraOpParallelismDatasetOp::kDatasetType,
                             MaxIntraOpParallelismDatasetOp::kDatasetOp})),
                        input, max_intra_op_parallelism);
}

void MaxIntraOpParallelismDatasetOp::MakeDataset(OpKernelContext* ctx,
                                                 DatasetBase* input,
                                                 DatasetBase** output) {
  int64_t max_intra_op_parallelism;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, "max_intra_op_parallelism",
                                              &max_intra_op_parallelism));
  OP_REQUIRES(
      ctx, max_intra_op_parallelism >= 0,
      errors::InvalidArgument("`max_intra_op_parallelism` must be >= 0"));
  *output = new Dataset(ctx, input, max_intra_op_parallelism);
}

class PrivateThreadPoolDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int num_threads)
      : Dataset(ctx, DatasetContext(ctx), input, num_threads) {}

  Dataset(OpKernelContext* ctx, DatasetContext&& dataset_ctx,
          const DatasetBase* input, int num_threads)
      : DatasetBase(std::move(dataset_ctx)),
        input_(input),
        num_threads_(num_threads == 0 ? port::MaxParallelism() : num_threads),
        traceme_metadata_(
            {{"num_threads",
              strings::Printf("%lld", static_cast<long long>(num_threads_))}}) {
    thread_pool_ = std::make_unique<thread::ThreadPool>(
        ctx->env(), ThreadOptions{}, "data_private_threadpool", num_threads_);
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::PrivateThreadPool")});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return "PrivateThreadPoolDatasetOp::Dataset";
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return input_->Cardinality(options);
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->clear();
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* num_threads_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(num_threads_, &num_threads_node));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, num_threads_node}, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    absl::Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      thread::ThreadPool* pool = dataset()->thread_pool_.get();
      IteratorContext::Params params(ctx);
      params.runner = [pool](std::function<void()> c) {
        pool->Schedule(std::move(c));
      };
      params.runner_threadpool_size = dataset()->num_threads_;
      return input_impl_->GetNext(IteratorContext{std::move(params)},
                                  out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      DCHECK(input_impl_ != nullptr);
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      return absl::OkStatus();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      return dataset()->traceme_metadata_;
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
  };

  const DatasetBase* const input_;
  const int64_t num_threads_;
  const TraceMeMetadata traceme_metadata_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

/* static */
void PrivateThreadPoolDatasetOp::MakeDatasetFromOptions(OpKernelContext* ctx,
                                                        DatasetBase* input,
                                                        int32_t num_threads,
                                                        DatasetBase** output) {
  OP_REQUIRES_OK(ctx, ValidateNumThreads(num_threads));
  *output = new Dataset(ctx,
                        DatasetContext(DatasetContext::Params(
                            {PrivateThreadPoolDatasetOp::kDatasetType,
                             PrivateThreadPoolDatasetOp::kDatasetOp})),
                        input, num_threads);
}

void PrivateThreadPoolDatasetOp::MakeDataset(OpKernelContext* ctx,
                                             DatasetBase* input,
                                             DatasetBase** output) {
  int64_t num_threads = 0;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument<int64_t>(ctx, "num_threads", &num_threads));
  OP_REQUIRES_OK(ctx, ValidateNumThreads(num_threads));
  *output = new Dataset(ctx, input, num_threads);
}

namespace {

REGISTER_KERNEL_BUILDER(Name("MaxIntraOpParallelismDataset").Device(DEVICE_CPU),
                        MaxIntraOpParallelismDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalMaxIntraOpParallelismDataset").Device(DEVICE_CPU),
    MaxIntraOpParallelismDatasetOp);

REGISTER_KERNEL_BUILDER(Name("PrivateThreadPoolDataset").Device(DEVICE_CPU),
                        PrivateThreadPoolDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalPrivateThreadPoolDataset").Device(DEVICE_CPU),
    PrivateThreadPoolDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ThreadPoolHandle").Device(DEVICE_CPU),
                        ThreadPoolHandleOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalThreadPoolHandle").Device(DEVICE_CPU),
                        ThreadPoolHandleOp);

REGISTER_KERNEL_BUILDER(Name("ThreadPoolDataset").Device(DEVICE_CPU),
                        ThreadPoolDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalThreadPoolDataset").Device(DEVICE_CPU),
    ThreadPoolDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
