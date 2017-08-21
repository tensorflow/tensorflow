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
#include "tensorflow/core/kernels/dataset.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following ops.

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " types but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != received[i]) {
      return errors::InvalidArgument("Data type mismatch at component ", i,
                                     ": expected ", DataTypeString(expected[i]),
                                     " but got ", DataTypeString(received[i]),
                                     ".");
    }
  }
  return Status::OK();
}

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " shapes but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!expected[i].IsCompatibleWith(received[i])) {
      return errors::InvalidArgument("Incompatible shapes at component ", i,
                                     ": expected ", expected[i].DebugString(),
                                     " but got ", received[i].DebugString(),
                                     ".");
    }
  }

  return Status::OK();
}

class IteratorResource : public ResourceBase {
 public:
  IteratorResource(const DataTypeVector& output_dtypes,
                   const std::vector<PartialTensorShape>& output_shapes)
      : iterator_(nullptr),
        output_dtypes_(output_dtypes),
        output_shapes_(output_shapes) {}

  Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) {
    std::shared_ptr<IteratorBase> captured_iterator(iterator_);
    if (captured_iterator) {
      return captured_iterator->GetNext(ctx, out_tensors, end_of_sequence);
    } else {
      return errors::FailedPrecondition(
          "GetNext() failed because the iterator has not been initialized. "
          "Ensure that you have run the initializer operation for this "
          "iterator before getting the next element.");
    }
  }

  // Transfers ownership of iterator to this. This method is thread-safe.
  Status set_iterator(std::unique_ptr<IteratorBase> iterator) {
    if (iterator) {
      TF_RETURN_IF_ERROR(
          VerifyTypesMatch(output_dtypes_, iterator->output_dtypes()));
      TF_RETURN_IF_ERROR(
          VerifyShapesCompatible(output_shapes_, iterator->output_shapes()));
    }
    iterator_.reset(iterator.release());
    return Status::OK();
  }

  string DebugString() override { return "Iterator resource"; }

  const DataTypeVector& output_dtypes() const { return output_dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const {
    return output_shapes_;
  }

 private:
  std::shared_ptr<IteratorBase> iterator_;
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
};

// TODO(mrry): Can we simply use the template kernel here?
class IteratorHandleOp : public ResourceOpKernel<IteratorResource> {
 public:
  explicit IteratorHandleOp(OpKernelConstruction* ctx)
      : ResourceOpKernel<IteratorResource>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 private:
  Status CreateResource(IteratorResource** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    *ret = new IteratorResource(output_dtypes_, output_shapes_);
    return Status::OK();
  }

  Status VerifyResource(IteratorResource* resource) override {
    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_dtypes_, resource->output_dtypes()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, resource->output_shapes()));
    return Status::OK();
  }

 private:
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

class MakeIteratorOp : public OpKernel {
 public:
  explicit MakeIteratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &dataset));
    core::ScopedUnref unref_dataset(dataset);
    IteratorResource* iterator_resource;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &iterator_resource));
    OP_REQUIRES_OK(ctx, iterator_resource->set_iterator(
                            dataset->MakeIterator("Iterator")));
    iterator_resource->Unref();
  }
};

class OneShotIteratorOp : public AsyncOpKernel {
 public:
  explicit OneShotIteratorOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        thread_pool_(new thread::ThreadPool(
            ctx->env(), ThreadOptions(),
            strings::StrCat("one_shot_iterator_initialization_thread_",
                            SanitizeThreadSuffix(name())),
            1 /* num_threads */, false /* low_latency_hint */))

  {
    string shared_name;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name));
    OP_REQUIRES(ctx, shared_name.empty(),
                errors::InvalidArgument("OneShotIteratorOp does not currently "
                                        "support the 'shared_name' attr."));
    const NameAttrList* dataset_factory_func;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dataset_factory", &dataset_factory_func));
    dataset_factory_func_ = *dataset_factory_func;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  ~OneShotIteratorOp() override {
    if (iterator_resource_ != nullptr) {
      iterator_resource_->Unref();
      if (!cinfo_.resource_manager()
               ->Delete<IteratorResource>(cinfo_.container(), cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

  // NOTE(mrry): This is based on `ResourceOpKernel<T>::Compute()`,
  // but due to the fact that `ResourceOpKernel<T>::CreateResource()`
  // does not provide access to the `OpKernelContext*` and we need
  // this to invoke the factory function, it's not possible to
  // implement this kernel by implementing `CreateResource()`.
  // Furthermore, due to the fact that this kernel might block when
  // running the initialization function, we must implement this
  // kernel as an async kernel.
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    {
      mutex_lock l(mu_);
      if (iterator_resource_ == nullptr && initialization_status_.ok()) {
        // The initialization thread will call `done`.
        if (!initialization_started_) {
          // TODO(mrry): Convert the initialization code to use
          // callbacks instead of wasting a thread.
          thread_pool_->Schedule([this, ctx, done]() { Init(ctx, done); });
          initialization_started_ = true;
        } else {
          done_callbacks_.emplace_back(ctx, std::move(done));
        }
        return;
      }
    }
    ProduceOutput(ctx, std::move(done));
  }

 private:
  void Init(OpKernelContext* ctx, DoneCallback done) {
    IteratorResource* iterator = nullptr;
    ContainerInfo cinfo;
    Status s = TryInit(ctx, &iterator, &cinfo);

    std::vector<std::pair<OpKernelContext*, DoneCallback>> callbacks_to_run;
    {
      mutex_lock l(mu_);
      if (s.ok()) {
        iterator_resource_ = iterator;
        cinfo_ = cinfo;
      }
      initialization_status_ = s;
      std::swap(done_callbacks_, callbacks_to_run);
    }

    for (auto&& ctx_done : callbacks_to_run) {
      ProduceOutput(ctx_done.first, std::move(ctx_done.second));
    }
    ProduceOutput(ctx, std::move(done));
  }

  Status TryInit(OpKernelContext* ctx, IteratorResource** iterator,
                 ContainerInfo* cinfo) {
    TF_RETURN_IF_ERROR(cinfo->Init(ctx->resource_manager(), def()));

    // Create an IteratorResource that will hold the iterator for this op.
    TF_RETURN_IF_ERROR(
        ctx->resource_manager()->LookupOrCreate<IteratorResource>(
            cinfo->container(), cinfo->name(), iterator,
            [this](IteratorResource** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              *ret = new IteratorResource(output_dtypes_, output_shapes_);
              return Status::OK();
            }));

    core::ScopedUnref unref_iterator(*iterator);

    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_dtypes_, (*iterator)->output_dtypes()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, (*iterator)->output_shapes()));

    // Call the dataset_factory_func_ to create a new dataset,
    // over which this op will iterate.
    FunctionLibraryRuntime::Handle f_handle;
    TF_RETURN_IF_ERROR(ctx->function_library()->Instantiate(
        dataset_factory_func_.name(), AttrSlice(&dataset_factory_func_.attr()),
        &f_handle));
    FunctionLibraryRuntime::Options opts;
    opts.cancellation_manager = ctx->cancellation_manager();
    // Choose a step ID that is guaranteed not to clash with any
    // Session-generated step ID. DirectSession only generates
    // non-negative step IDs (contiguous, starting from 0), and
    // MasterSession generates 56-bit random step IDs whose MSB is
    // always 0, so a negative random step ID should suffice.
    opts.step_id = -std::abs(static_cast<int64>(random::New64()));
    ScopedStepContainer step_container(opts.step_id, [ctx](const string& name) {
      ctx->resource_manager()->Cleanup(name).IgnoreError();
    });
    opts.step_container = &step_container;
    opts.runner = ctx->runner();
    Notification n;
    Status factory_status;
    std::vector<Tensor> return_values;
    ctx->function_library()->Run(opts, f_handle, {}, &return_values,
                                 [&n, &factory_status](Status s) {
                                   factory_status.Update(s);
                                   n.Notify();
                                 });
    n.WaitForNotification();
    TF_RETURN_IF_ERROR(factory_status);
    if (return_values.size() != 1 || return_values[0].dtype() != DT_RESOURCE ||
        !TensorShapeUtils::IsScalar(return_values[0].shape())) {
      return errors::InvalidArgument(
          "The `dataset_factory` function must return "
          "a single scalar of dtype DT_RESOURCE.");
    }

    // Retrieve the dataset that was created in the factory function.
    DatasetBase* dataset;
    const ResourceHandle& dataset_resource =
        return_values[0].flat<ResourceHandle>()(0);
    TF_RETURN_IF_ERROR(LookupResource(ctx, dataset_resource, &dataset));
    core::ScopedUnref unref_dataset(dataset);

    // Create an iterator for the dataset that was created in the
    // factory function. This transfers ownership of the dataset to
    // the iterator, so we can delete it from the resource manager.
    TF_RETURN_IF_ERROR(
        (*iterator)->set_iterator(dataset->MakeIterator("Iterator")));
    TF_RETURN_IF_ERROR(DeleteResource<DatasetBase>(ctx, dataset_resource));

    (*iterator)->Ref();
    return Status::OK();
  }

  void ProduceOutput(OpKernelContext* ctx, DoneCallback done) {
    Tensor* handle;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, TensorShape({}), &handle),
                         done);
    Status s;
    {
      mutex_lock l(mu_);
      s = initialization_status_;
      if (s.ok()) {
        handle->scalar<ResourceHandle>()() =
            MakeResourceHandle<IteratorResource>(ctx, cinfo_.container(),
                                                 cinfo_.name());
      }
    }
    OP_REQUIRES_OK_ASYNC(ctx, s, done);
    done();
  }

  NameAttrList dataset_factory_func_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;

  std::unique_ptr<thread::ThreadPool> thread_pool_;

  mutex mu_;
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  IteratorResource* iterator_resource_ GUARDED_BY(mu_) = nullptr;

  bool initialization_started_ GUARDED_BY(mu_) = false;
  Status initialization_status_ GUARDED_BY(mu_);
  std::vector<std::pair<OpKernelContext*, DoneCallback>> done_callbacks_
      GUARDED_BY(mu_);
};

class IteratorGetNextOp : public AsyncOpKernel {
 public:
  explicit IteratorGetNextOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        thread_pool_(new thread::ThreadPool(
            ctx->env(), ThreadOptions(),
            strings::StrCat("iterator_get_next_thread_",
                            SanitizeThreadSuffix(name())),
            1 /* num_threads */, false /* low_latency_hint */)) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    IteratorResource* iterator;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &iterator));

    // The call to `iterator->GetNext()` may block and depend on an
    // inter-op thread pool thread, so we issue the call from the
    // owned thread pool.
    thread_pool_->Schedule([this, ctx, iterator, done]() {
      core::ScopedUnref unref_iterator(iterator);

      std::vector<Tensor> components;
      bool end_of_sequence = false;

      IteratorContext::Params params;
      params.env = ctx->env();
      params.step_id = ctx->step_id();
      params.resource_manager = ctx->resource_manager();
      params.runner = *(ctx->runner());
      IteratorContext iter_ctx(std::move(params));

      OP_REQUIRES_OK_ASYNC(
          ctx, iterator->GetNext(&iter_ctx, &components, &end_of_sequence),
          done);
      OP_REQUIRES_ASYNC(ctx, !end_of_sequence,
                        errors::OutOfRange("End of sequence"), done);

      for (int i = 0; i < components.size(); ++i) {
        // TODO(mrry): Check that the shapes match the shape attrs.
        ctx->set_output(i, components[i]);
      }

      done();
    });
  }

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class IteratorDisposeOp : public OpKernel {
 public:
  explicit IteratorDisposeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    IteratorResource* iterator;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &iterator));
    core::ScopedUnref unref_iterator(iterator);
    OP_REQUIRES_OK(ctx, iterator->set_iterator(nullptr));
  }
};

class IteratorToStringHandleOp : public OpKernel {
 public:
  explicit IteratorToStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& resource_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
                errors::InvalidArgument("resource_handle must be a scalar"));

    // Validate that the handle corresponds to a real resource, and
    // that it is an IteratorResource.
    IteratorResource* iterator_resource;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator_resource));
    iterator_resource->Unref();

    Tensor* string_handle_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &string_handle_t));
    string_handle_t->scalar<string>()() =
        resource_handle_t.scalar<ResourceHandle>()().SerializeAsString();
  }
};

class IteratorFromStringHandleOp : public OpKernel {
 public:
  explicit IteratorFromStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES(
        ctx,
        output_dtypes_.empty() || output_shapes_.empty() ||
            output_dtypes_.size() == output_shapes_.size(),
        errors::InvalidArgument("If both 'output_types' and 'output_shapes' "
                                "are set, they must have the same length."));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& string_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(string_handle_t.shape()),
                errors::InvalidArgument("string_handle must be a scalar"));

    ResourceHandle resource_handle;
    OP_REQUIRES(
        ctx,
        resource_handle.ParseFromString(string_handle_t.scalar<string>()()),
        errors::InvalidArgument(
            "Could not parse string_handle as a valid ResourceHandle"));

    OP_REQUIRES(
        ctx, resource_handle.device() == ctx->device()->attributes().name(),
        errors::InvalidArgument("Attempted create an iterator on device \"",
                                ctx->device()->attributes().name(),
                                "\" from handle defined on device \"",
                                resource_handle.device(), "\""));

    // Validate that the handle corresponds to a real resource, and
    // that it is an IteratorResource.
    IteratorResource* iterator_resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, resource_handle, &iterator_resource));
    core::ScopedUnref unref_iterator(iterator_resource);
    if (!output_dtypes_.empty()) {
      OP_REQUIRES_OK(ctx, VerifyTypesMatch(output_dtypes_,
                                           iterator_resource->output_dtypes()));
    }
    if (!output_shapes_.empty()) {
      OP_REQUIRES_OK(
          ctx, VerifyShapesCompatible(output_shapes_,
                                      iterator_resource->output_shapes()));
    }

    Tensor* resource_handle_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &resource_handle_t));
    resource_handle_t->scalar<ResourceHandle>()() = resource_handle;
  }

 private:
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("Iterator").Device(DEVICE_CPU), IteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("MakeIterator").Device(DEVICE_CPU),
                        MakeIteratorOp);
REGISTER_KERNEL_BUILDER(Name("OneShotIterator").Device(DEVICE_CPU),
                        OneShotIteratorOp);
REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE_CPU),
                        IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(Name("IteratorDispose").Device(DEVICE_CPU),
                        IteratorDisposeOp);
REGISTER_KERNEL_BUILDER(Name("IteratorToStringHandle").Device(DEVICE_CPU),
                        IteratorToStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorFromStringHandle").Device(DEVICE_CPU),
                        IteratorFromStringHandleOp);

}  // namespace

}  // namespace tensorflow
