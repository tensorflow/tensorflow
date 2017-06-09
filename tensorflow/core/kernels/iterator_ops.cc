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
    OP_REQUIRES_OK(ctx,
                   iterator_resource->set_iterator(dataset->MakeIterator()));
    iterator_resource->Unref();
  }
};

class OneShotIteratorOp : public OpKernel {
 public:
  explicit OneShotIteratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string shared_name;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name));
    OP_REQUIRES(ctx, shared_name.empty(),
                errors::InvalidArgument("OneShotIteratorOp does not currently "
                                        "support the 'shared_name' attr."));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("dataset_factory", &dataset_factory_func_));
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
  // does not provide access to the `OpKernelContext*` and we need this
  // to invoke the factory function, it's not possible to implement
  // this kernel by implementing `CreateResource()`.
  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    if (iterator_resource_ == nullptr) {
      ResourceMgr* mgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, cinfo_.Init(mgr, def()));

      // Create an IteratorResource that will hold the iterator for this op.
      IteratorResource* resource;
      OP_REQUIRES_OK(
          ctx,
          mgr->LookupOrCreate<IteratorResource>(
              cinfo_.container(), cinfo_.name(), &resource,
              [this](IteratorResource** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                *ret = new IteratorResource(output_dtypes_, output_shapes_);
                return Status::OK();
              }));
      Status s = VerifyTypesMatch(output_dtypes_, resource->output_dtypes());
      s.Update(
          VerifyShapesCompatible(output_shapes_, resource->output_shapes()));
      if (TF_PREDICT_FALSE(!s.ok())) {
        resource->Unref();
        ctx->SetStatus(s);
        return;
      }
      iterator_resource_ = resource;

      // Call the dataset_factory_func_ to create a new dataset,
      // over which this op will iterate.
      FunctionLibraryRuntime::Handle f_handle;
      OP_REQUIRES_OK(ctx,
                     ctx->function_library()->Instantiate(
                         dataset_factory_func_->name(),
                         AttrSlice(&dataset_factory_func_->attr()), &f_handle));
      FunctionLibraryRuntime::Options opts;
      opts.cancellation_manager = ctx->cancellation_manager();
      // Choose a step ID that is guaranteed not to clash with any
      // Session-generated step ID. DirectSession only generates
      // non-negative step IDs (contiguous, starting from 0), and
      // MasterSession generates 56-bit random step IDs whose MSB is
      // always 0, so a negative random step ID should suffice.
      opts.step_id = -std::abs(static_cast<int64>(random::New64()));
      ScopedStepContainer step_container(
          opts.step_id, [ctx](const string& name) {
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
      OP_REQUIRES_OK(ctx, factory_status);
      OP_REQUIRES(
          ctx,
          return_values.size() == 1 &&
              return_values[0].dtype() == DT_RESOURCE &&
              TensorShapeUtils::IsScalar(return_values[0].shape()),
          errors::InvalidArgument("The `dataset_factory` function must return "
                                  "a single scalar of dtype DT_RESOURCE."));

      // Retrieve the dataset that was created in the factory function.
      DatasetBase* dataset;
      const ResourceHandle& dataset_resource =
          return_values[0].flat<ResourceHandle>()(0);
      OP_REQUIRES_OK(ctx, LookupResource(ctx, dataset_resource, &dataset));
      core::ScopedUnref unref_dataset(dataset);

      // Create an iterator for the dataset that was created in the
      // factory function. This transfers ownership of the dataset to
      // the iterator, so we can delete it from the resource manager.
      OP_REQUIRES_OK(ctx,
                     iterator_resource_->set_iterator(dataset->MakeIterator()));
      OP_REQUIRES_OK(ctx, DeleteResource<DatasetBase>(ctx, dataset_resource));
    }
    Tensor* handle;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    handle->scalar<ResourceHandle>()() = MakeResourceHandle<IteratorResource>(
        ctx, cinfo_.container(), cinfo_.name());
  }

 private:
  const NameAttrList* dataset_factory_func_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;

  mutex mu_;
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  IteratorResource* iterator_resource_ = nullptr;
};

class IteratorGetNextOp : public AsyncOpKernel {
 public:
  explicit IteratorGetNextOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        thread_pool_(new thread::ThreadPool(
            ctx->env(), ThreadOptions(),
            strings::StrCat("iterator_get_next_thread_",
                            SanitizeThreadSuffix(def().name())),
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

REGISTER_KERNEL_BUILDER(Name("Iterator").Device(DEVICE_CPU), IteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("MakeIterator").Device(DEVICE_CPU),
                        MakeIteratorOp);
REGISTER_KERNEL_BUILDER(Name("OneShotIterator").Device(DEVICE_CPU),
                        OneShotIteratorOp);
REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE_CPU),
                        IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(Name("IteratorDispose").Device(DEVICE_CPU),
                        IteratorDisposeOp);

}  // namespace

}  // namespace tensorflow
