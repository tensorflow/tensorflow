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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"

namespace tensorflow {
namespace data {

class IteratorResource;

class IteratorHandleOp : public OpKernel {
 public:
  explicit IteratorHandleOp(OpKernelConstruction* ctx);

  // The resource is deleted from the resource manager only when it is private
  // to kernel. Ideally the resource should be deleted when it is no longer held
  // by anyone, but it would break backward compatibility.
  ~IteratorHandleOp() override;

  void Compute(OpKernelContext* context) override LOCKS_EXCLUDED(mu_);

 private:
  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  Status VerifyResource(IteratorResource* resource);

  template <typename To, typename From>  // use like this: down_cast<T*>(foo);
  static inline To down_cast(From* f) {  // so we only accept pointers
    static_assert(
        (std::is_base_of<From, typename std::remove_pointer<To>::type>::value),
        "target type not derived from source type");

    // We skip the assert and hence the dynamic_cast if RTTI is disabled.
#if !defined(__GNUC__) || defined(__GXX_RTTI)
    // Uses RTTI in dbg and fastbuild. asserts are disabled in opt builds.
    assert(f == nullptr || dynamic_cast<To>(f) != nullptr);
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)
    return static_cast<To>(f);
  }

  FunctionLibraryRuntime* CreatePrivateFLR(
      OpKernelContext* ctx, std::unique_ptr<DeviceMgr>* device_mgr,
      std::unique_ptr<FunctionLibraryDefinition>* flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime>* pflr);

  mutex mu_;
  ContainerInfo cinfo_;  // Written once under mu_ then constant afterwards.
  IteratorResource* resource_ GUARDED_BY(mu_) = nullptr;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  const int graph_def_version_;
  string name_;
};

// Like IteratorHandleOp, but creates handles which are never shared, and does
// not hold a reference to these handles. The latter is important for eager
// execution, since OpKernel instances generally live as long as the program
// running them.
class AnonymousIteratorHandleOp : public OpKernel {
 public:
  explicit AnonymousIteratorHandleOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* context) override;

 private:
  // Coordinates Iterator unique name creation across AnonymousIteratorHandleOp
  // instances.
  static mutex static_resource_lookup_mutex_;
  // current_id_ is just a hint for creating unique names. If it turns out
  // there's a collision (e.g. because another AnonymousIteratorHandleOp
  // instance is generating handles) we'll just skip that id.
  static int64 current_id_ GUARDED_BY(static_resource_lookup_mutex_);
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  const int graph_def_version_;
};

class MakeIteratorOp : public OpKernel {
 public:
  explicit MakeIteratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class IteratorGetNextOp : public AsyncOpKernel {
 public:
  explicit IteratorGetNextOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "tf_data_iterator_get_next") {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  BackgroundWorker background_worker_;
};

class IteratorGetNextSyncOp : public OpKernel {
 public:
  explicit IteratorGetNextSyncOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
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

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_ITERATOR_OPS_H_
