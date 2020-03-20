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

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/unbounded_thread_pool.h"
#include "tensorflow/core/kernels/ops_util.h"

namespace tensorflow {
namespace data {

class IteratorResource : public ResourceBase {
 public:
  IteratorResource(Env* env, const DataTypeVector& output_dtypes,
                   const std::vector<PartialTensorShape>& output_shapes,
                   const int /*unused: graph_def_version*/,
                   std::unique_ptr<DeviceMgr> device_mgr,
                   std::unique_ptr<FunctionLibraryDefinition> flib_def,
                   std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                   FunctionLibraryRuntime* flr)
      : unbounded_thread_pool_(env, "tf_data_iterator_resource"),
        device_mgr_(std::move(device_mgr)),
        iterator_state_(std::make_shared<State>(std::move(flib_def),
                                                std::move(pflr), flr,
                                                /*iterator=*/nullptr)),
        output_dtypes_(output_dtypes),
        output_shapes_(output_shapes) {
    VLOG(2) << "constructor";
  }

  ~IteratorResource() override { VLOG(2) << "destructor"; }

  Status GetNext(OpKernelContext* ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence);

  Status Save(SerializationContext* ctx, IteratorStateWriter* writer);

  Status Restore(OpKernelContext* ctx, IteratorStateReader* reader);

  Status SetIteratorFromDataset(OpKernelContext* ctx, DatasetBase* dataset);

  string DebugString() const override { return "Iterator resource"; }

  const DataTypeVector& output_dtypes() const { return output_dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const {
    return output_shapes_;
  }

 private:
  // TODO(aaudibert): convert to a class for better encapsulation.
  struct State {
    State(std::shared_ptr<FunctionLibraryDefinition> flib_def,
          std::shared_ptr<ProcessFunctionLibraryRuntime> pflr,
          FunctionLibraryRuntime* flr,
          std::unique_ptr<DatasetBaseIterator> iterator)
        : flib_def(std::move(flib_def)),
          flr(flr),
          pflr(std::move(pflr)),
          function_handle_cache(absl::make_unique<FunctionHandleCache>(flr)),
          iterator(std::move(iterator)) {}

    ~State() { cancellation_manager.StartCancel(); }

    // Downcasts the given `IteratorBase` to a `DatasetBaseIterator`, and uses
    // it to set the `iterator` field.
    void DowncastAndSetIterator(std::unique_ptr<IteratorBase> it) {
      iterator.reset(static_cast<DatasetBaseIterator*>(it.release()));
    }

    std::shared_ptr<FunctionLibraryDefinition> flib_def;
    FunctionLibraryRuntime* flr = nullptr;  // not owned.
    std::shared_ptr<ProcessFunctionLibraryRuntime> pflr;
    std::unique_ptr<FunctionHandleCache> function_handle_cache;
    ResourceMgr resource_mgr;
    CancellationManager cancellation_manager;
    std::unique_ptr<DatasetBaseIterator> iterator;
  };

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
      : HybridAsyncOpKernel(ctx, "tf_data_iterator_get_next") {}

  AsyncOpKernel* AsAsync() override;

 protected:
  Status DoCompute(OpKernelContext* ctx) override;
};

class DeleteIteratorOp : public OpKernel {
 public:
  explicit DeleteIteratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
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
