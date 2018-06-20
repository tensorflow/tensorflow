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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace {

class ThreadPoolResource : public ResourceBase {
 public:
  ThreadPoolResource(Env* env, const ThreadOptions& thread_options,
                     const string& name, int num_threads, bool low_latency_hint)
      : thread_pool_(env, thread_options, name, num_threads, low_latency_hint) {
  }

  // Schedules fn() for execution in the pool of threads.
  void Schedule(std::function<void()> fn) {
    thread_pool_.Schedule(std::move(fn));
  }

  string DebugString() override { return "ThreadPoolResource"; }

 private:
  thread::ThreadPool thread_pool_;
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
                                        false /* low_latency_hint */);
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
};

class ThreadPoolDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ThreadPoolDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    ThreadPoolResource* threadpool_resource;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1),
                                       &threadpool_resource));
    core::ScopedUnref unref_iterator(threadpool_resource);

    *output = new Dataset(ctx, input, threadpool_resource);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            ThreadPoolResource* threadpool)
        : GraphDatasetBase(ctx), input_(input), threadpool_(threadpool) {
      input_->Ref();
      threadpool_->Ref();
    }

    ~Dataset() override {
      input_->Unref();
      threadpool_->Unref();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::ThreadPool")}));
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

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      return errors::Unimplemented(
          "Cannot currently serialize the thread pool for a "
          "ThreadPoolDataset.");
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
        ThreadPoolResource* pool = dataset()->threadpool_;
        IteratorContext::Params params;
        params.env = ctx->env();
        params.runner = [pool](std::function<void()> c) {
          pool->Schedule(std::move(c));
        };
        params.stats_aggregator_getter = ctx->stats_aggregator_getter();
        params.lib = ctx->lib();
        params.function_library = ctx->function_library();
        params.allocator_getter = ctx->allocator_getter();
        IteratorContext threadpool_ctx(params);
        return input_impl_->GetNext(&threadpool_ctx, out_tensors,
                                    end_of_sequence);
      }

     private:
      std::unique_ptr<IteratorBase> input_impl_;
    };

    const DatasetBase* const input_;
    ThreadPoolResource* const threadpool_;
  };
};

REGISTER_KERNEL_BUILDER(Name("ThreadPoolHandle").Device(DEVICE_CPU),
                        ThreadPoolHandleOp);
REGISTER_KERNEL_BUILDER(Name("ThreadPoolDataset").Device(DEVICE_CPU),
                        ThreadPoolDatasetOp);

}  // namespace
}  // namespace tensorflow
