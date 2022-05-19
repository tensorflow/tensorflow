/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_ITERATOR_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_ITERATOR_OPS_H_

#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/metric_utils.h"
#include "tensorflow/core/data/unbounded_thread_pool.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/refcount.h"

namespace tensorflow {
namespace data {

class IteratorResource : public ResourceBase {
 public:
  IteratorResource(Env* env, const DataTypeVector& output_dtypes,
                   const std::vector<PartialTensorShape>& output_shapes,
                   std::unique_ptr<DeviceMgr> device_mgr,
                   std::unique_ptr<FunctionLibraryDefinition> flib_def,
                   std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                   FunctionLibraryRuntime* flr);

  ~IteratorResource() override;

  // Gets the next output from the iterator managed by this iterator resource.
  //
  // If at least one output remains, that output will be stored in
  // `*out_tensors` and `false` will be stored in `*end_of_sequence`.
  //
  // If no more outputs remain, `true` will be stored in `*end_of_sequence`, and
  // the content of `*out_tensors` will be undefined.
  Status GetNext(OpKernelContext* ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence);

  // Saves a checkpoint of the state of the iterator through the given `writer`.
  Status Save(SerializationContext* ctx, IteratorStateWriter* writer);

  // Restores the state of the iterator from a checkpoint created by `Save`.
  Status Restore(OpKernelContext* ctx, IteratorStateReader* reader);

  // Creates an iterator for `dataset`, and associates the iterator with this
  // iterator resource.
  //
  // `SetIteratorFromDataset` should be called before calling `GetNext`, `Save`,
  // or `Restore`.
  Status SetIteratorFromDataset(OpKernelContext* ctx,
                                const DatasetBase* dataset);

  string DebugString() const override { return "Iterator resource"; }

  const DataTypeVector& output_dtypes() const { return output_dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const {
    return output_shapes_;
  }

 private:
  class State {
   public:
    State(std::shared_ptr<FunctionLibraryDefinition> flib_def,
          std::shared_ptr<ProcessFunctionLibraryRuntime> pflr,
          FunctionLibraryRuntime* flr,
          std::unique_ptr<DatasetBaseIterator> iterator)
        : flib_def_(std::move(flib_def)),
          flr_(flr),
          pflr_(std::move(pflr)),
          function_handle_cache_(absl::make_unique<FunctionHandleCache>(flr)),
          iterator_(std::move(iterator)) {}

    ~State() { cancellation_manager_.StartCancel(); }

    // Downcasts the given `IteratorBase` to a `DatasetBaseIterator`, and uses
    // it to set the `iterator` and the `dataset` field.
    void DowncastAndSetIteratorAndDataset(std::unique_ptr<IteratorBase> it,
                                          const DatasetBase* dataset) {
      iterator_.reset(static_cast<DatasetBaseIterator*>(it.release()));
      if (dataset) {
        dataset->Ref();
        dataset_.reset(const_cast<DatasetBase*>(dataset));
      }
    }

    std::shared_ptr<FunctionLibraryDefinition> flib_def() { return flib_def_; }

    FunctionLibraryRuntime* flr() { return flr_; }

    std::shared_ptr<ProcessFunctionLibraryRuntime> pflr() { return pflr_; }

    FunctionHandleCache* function_handle_cache() {
      return function_handle_cache_.get();
    }

    ResourceMgr* resource_mgr() { return &resource_mgr_; }

    CancellationManager* cancellation_manager() {
      return &cancellation_manager_;
    }

    DatasetBaseIterator* iterator() { return iterator_.get(); }

    DatasetBase* dataset() { return dataset_.get(); }

   private:
    std::shared_ptr<FunctionLibraryDefinition> flib_def_;
    FunctionLibraryRuntime* flr_ = nullptr;  // not owned
    std::shared_ptr<ProcessFunctionLibraryRuntime> pflr_;
    std::unique_ptr<FunctionHandleCache> function_handle_cache_;
    ResourceMgr resource_mgr_;
    CancellationManager cancellation_manager_;
    std::unique_ptr<DatasetBaseIterator> iterator_;
    core::RefCountPtr<DatasetBase> dataset_;
  };

  IteratorMetricsCollector metrics_collector_;
  UnboundedThreadPool unbounded_thread_pool_;

  mutex mu_;
  const std::unique_ptr<DeviceMgr> device_mgr_ TF_GUARDED_BY(mu_);
  std::shared_ptr<State> iterator_state_ TF_GUARDED_BY(mu_);
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
};

class IteratorHandleOp : public OpKernel {
 public:
  explicit IteratorHandleOp(OpKernelConstruction* ctx);

  // The resource is deleted from the resource manager only when it is private
  // to kernel. Ideally the resource should be deleted when it is no longer held
  // by anyone, but it would break backward compatibility.
  ~IteratorHandleOp() override;

  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_);

 private:
  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  Status VerifyResource(IteratorResource* resource);

  FunctionLibraryRuntime* CreatePrivateFLR(
      OpKernelContext* ctx, std::unique_ptr<DeviceMgr>* device_mgr,
      std::unique_ptr<FunctionLibraryDefinition>* flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime>* pflr);

  mutex mu_;
  ContainerInfo cinfo_;  // Written once under mu_ then constant afterwards.
  IteratorResource* resource_ TF_GUARDED_BY(mu_) = nullptr;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  const int graph_def_version_;
  string name_;
};

// Like IteratorHandleOp, but creates handles which are never shared, and does
// not hold a reference to these handles. The latter is important for eager
// execution, since OpKernel instances generally live as long as the program
// running them.
class AnonymousIteratorHandleOp : public AnonymousResourceOp<IteratorResource> {
 public:
  explicit AnonymousIteratorHandleOp(OpKernelConstruction* context);

 private:
  string name() override;

  Status CreateResource(OpKernelContext* ctx,
                        std::unique_ptr<FunctionLibraryDefinition> flib_def,
                        std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                        FunctionLibraryRuntime* lib,
                        IteratorResource** resource) override;

  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  const int graph_def_version_;
};

// A hybrid asynchronous-and-synchronous OpKernel with efficient support for
// both modes.
//
// Inherit from this class when the application logic of the kernel (i) is
// implemented synchronously, (ii) must run on a background thread when the
// kernel executes in the inter-op threadpool (typically because it depends on
// inter-op threadpool threads, e.g. for function execution), and (iii) can run
// synchronously on the calling thread when the caller donates a thread
// (typically in eager execution). The implementation avoids a thread-hop in
// case (iii).
//
// NOTE: Unlike typical OpKernel subclasses, the application logic is
// implemented in a method (DoCompute()) that returns Status. Use
// TF_RETURN_IF_ERROR for error-related control flow rather than
// OP_REQUIRES_OK().
class HybridAsyncOpKernel : public AsyncOpKernel {
 public:
  HybridAsyncOpKernel(OpKernelConstruction* ctx,
                      const char* background_worker_name);

  void Compute(OpKernelContext* ctx) final;
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) final;

 protected:
  virtual Status DoCompute(OpKernelContext* ctx) = 0;

 private:
  BackgroundWorker background_worker_;
};

class MakeIteratorOp : public HybridAsyncOpKernel {
 public:
  explicit MakeIteratorOp(OpKernelConstruction* ctx)
      : HybridAsyncOpKernel(ctx, "tf_data_make_iterator") {}

 protected:
  Status DoCompute(OpKernelContext* ctx) override;
};

class IteratorGetNextOp : public HybridAsyncOpKernel {
 public:
  explicit IteratorGetNextOp(OpKernelConstruction* ctx)
      : HybridAsyncOpKernel(ctx, "tf_data_iterator_get_next") {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  AsyncOpKernel* AsAsync() override;

 protected:
  Status DoCompute(OpKernelContext* ctx) override;

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class DeleteIteratorOp : public HybridAsyncOpKernel {
 public:
  explicit DeleteIteratorOp(OpKernelConstruction* ctx)
      : HybridAsyncOpKernel(ctx, "tf_data_delete_iterator") {}

 protected:
  Status DoCompute(OpKernelContext* ctx) override;
};

class IteratorGetNextAsOptionalOp : public HybridAsyncOpKernel {
 public:
  explicit IteratorGetNextAsOptionalOp(OpKernelConstruction* ctx)
      : HybridAsyncOpKernel(ctx, "tf_data_iterator_get_next_as_optional") {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  Status DoCompute(OpKernelContext* ctx) override;

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class IteratorToStringHandleOp : public OpKernel {
 public:
  explicit IteratorToStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class IteratorFromStringHandleOp : public OpKernel {
 public:
  explicit IteratorFromStringHandleOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

class SerializeIteratorOp : public OpKernel {
 public:
  static constexpr const char* const kExternalStatePolicy =
      "external_state_policy";

  explicit SerializeIteratorOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  SerializationContext::ExternalStatePolicy external_state_policy_ =
      SerializationContext::ExternalStatePolicy::kWarn;
};

class DeserializeIteratorOp : public OpKernel {
 public:
  explicit DeserializeIteratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_ITERATOR_OPS_H_
