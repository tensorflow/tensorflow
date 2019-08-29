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
#include <memory>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

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
    OP_REQUIRES(
        ctx, num_threads_ > 0,
        errors::InvalidArgument("`num_threads` must be greater than zero."));
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

  void Compute(OpKernelContext* ctx) override LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (!initialized_) {
      ResourceMgr* mgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, cinfo_.Init(mgr, def()));
      ThreadPoolResource* resource;
      OP_REQUIRES_OK(ctx, mgr->LookupOrCreate<ThreadPoolResource>(
                              cinfo_.container(), cinfo_.name(), &resource,
                              [this, ctx](ThreadPoolResource** ret)
                                  EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                                    *ret = new ThreadPoolResource(
                                        ctx->env(), {}, display_name_,
                                        num_threads_,
                                        /*low_latency_hint=*/false,
                                        max_intra_op_parallelism_);
                                    return Status::OK();
                                  }));
      initialized_ = true;
    }
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            MakeTypeIndex<ThreadPoolResource>()));
  }

 private:
  mutex mu_;
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  bool initialized_ GUARDED_BY(mu_) = false;
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
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1),
                                       &threadpool_resource));
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
      return absl::make_unique<Iterator>(
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

    int64 Cardinality() const override { return input_->Cardinality(); }

    Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* resource_handle_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddTensor(resource_handle_, &resource_handle_node));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_graph_node, resource_handle_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(
            IteratorContext(CreateParams(ctx)), prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
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

class MaxIntraOpParallelismDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit MaxIntraOpParallelismDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 max_intra_op_parallelism;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "max_intra_op_parallelism",
                                              &max_intra_op_parallelism));
    OP_REQUIRES(
        ctx, max_intra_op_parallelism >= 0,
        errors::InvalidArgument("`max_intra_op_parallelism` must be >= 0"));
    *output = new Dataset(ctx, input, max_intra_op_parallelism);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            int64 max_intra_op_parallelism)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          max_intra_op_parallelism_(max_intra_op_parallelism) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(Iterator::Params{
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

    int64 Cardinality() const override { return input_->Cardinality(); }

    Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* max_intra_op_parallelism_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(max_intra_op_parallelism_,
                                      &max_intra_op_parallelism_node));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_graph_node, max_intra_op_parallelism_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        IteratorContext::Params params(ctx);
        auto max_parallelism = dataset()->max_intra_op_parallelism_;
        params.runner =
            RunnerWithMaxParallelism(*ctx->runner(), max_parallelism);
        return input_impl_->GetNext(IteratorContext{std::move(params)},
                                    out_tensors, end_of_sequence);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

     private:
      std::unique_ptr<IteratorBase> input_impl_;
    };

    const DatasetBase* const input_;
    const int64 max_intra_op_parallelism_;
  };
};

class PrivateThreadPoolDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit PrivateThreadPoolDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 num_threads = 0;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "num_threads", &num_threads));
    OP_REQUIRES(ctx, num_threads >= 1,
                errors::InvalidArgument("`num_threads` must be >= 1"));
    *output = new Dataset(ctx, input, num_threads);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input, int num_threads)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          num_threads_(num_threads) {
      thread_pool_ = absl::make_unique<thread::ThreadPool>(
          ctx->env(), ThreadOptions{}, "data_private_threadpool", num_threads,
          /*low_latency_hint=*/false);
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(Iterator::Params{
          this, strings::StrCat(prefix, "::PrivateThreadPool")});
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

    int64 Cardinality() const override { return input_->Cardinality(); }

    Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* num_threads_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(num_threads_, &num_threads_node));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph_node, num_threads_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
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
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

     private:
      std::unique_ptr<IteratorBase> input_impl_;
    };

    const DatasetBase* const input_;
    const int64 num_threads_;
    std::unique_ptr<thread::ThreadPool> thread_pool_;
  };
};

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
