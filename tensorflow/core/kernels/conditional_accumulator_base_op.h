/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_OP_H_
#define TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_OP_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conditional_accumulator_base.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

typedef std::function<void()> DoneCallback;

namespace tensorflow {

/**
 * Defines a ConditionalAccumulatorBaseOp, which constructs a
 * ConditionalAccumulatorBase (via sub-class's Creator) and returns its handle.
 */
class ConditionalAccumulatorBaseOp : public OpKernel {
 public:
  explicit ConditionalAccumulatorBaseOp(OpKernelConstruction* context)
      : OpKernel(context), accumulator_handle_set_(false) {
    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                &accumulator_handle_, nullptr));
    OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    if (!accumulator_handle_set_) {
      OP_REQUIRES_OK(ctx, SetAccumulatorHandle(ctx));
    }
    ctx->set_output_ref(0, &mu_, accumulator_handle_.AccessTensor(ctx));
  }

 protected:
  ~ConditionalAccumulatorBaseOp() override {
    // If the accumulator object was not shared, delete it.
    if (accumulator_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      TF_CHECK_OK((cinfo_.resource_manager()
                       ->template Delete<ConditionalAccumulatorBase>(
                           cinfo_.container(), cinfo_.name())));
    }
  }

 protected:
  typedef std::function<Status(ConditionalAccumulatorBase**)> Creator;

  // Subclasses must override this
  virtual Creator GetCreator() const = 0;

  // Variables required to construct ConditionalAccumulator
  DataType dtype_;
  PartialTensorShape shape_;
  ContainerInfo cinfo_;

 private:
  Status SetAccumulatorHandle(OpKernelContext* ctx)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));

    // Check input signature
    DataTypeVector expected_inputs = {};
    TF_RETURN_IF_ERROR(ctx->MatchSignature(expected_inputs, {DT_STRING_REF}));

    Creator creator = GetCreator();
    ConditionalAccumulatorBase* accumulator;
    TF_RETURN_IF_ERROR(
        (cinfo_.resource_manager()
             ->template LookupOrCreate<ConditionalAccumulatorBase>(
                 cinfo_.container(), cinfo_.name(), &accumulator, creator)));
    core::ScopedUnref unref_me(accumulator);

    // Verify that the shared accumulator is compatible
    // with the requested arguments.
    TF_RETURN_IF_ERROR(accumulator->MatchesNodeDef(def()));
    auto h = accumulator_handle_.AccessTensor(ctx)->template flat<string>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    accumulator_handle_set_ = true;
    return Status::OK();
  }

  mutex mu_;
  PersistentTensor accumulator_handle_ GUARDED_BY(mu_);
  bool accumulator_handle_set_ GUARDED_BY(mu_);
};

/**
 * General OpKernel for ConditionalAccumulatorBase-related ops.
 */
class ConditionalAccumulatorBaseAsyncOpKernel : public AsyncOpKernel {
 public:
  explicit ConditionalAccumulatorBaseAsyncOpKernel(
      OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback callback) final {
    ConditionalAccumulatorBase* accumulator;
    OP_REQUIRES_OK_ASYNC(
        ctx, GetResourceFromContext(ctx, "handle", &accumulator), callback);
    ComputeAsync(ctx, accumulator, [callback, accumulator]() {
      accumulator->Unref();
      callback();
    });
  }

 protected:
  virtual void ComputeAsync(OpKernelContext* ctx,
                            ConditionalAccumulatorBase* accumulator,
                            DoneCallback callback) = 0;
};

/**
 * General OpKernel for ConditionalAccumulatorBase-related ops.
 */
class ConditionalAccumulatorBaseSyncOpKernel : public OpKernel {
 public:
  explicit ConditionalAccumulatorBaseSyncOpKernel(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) final {
    ConditionalAccumulatorBase* accumulator;
    OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "handle", &accumulator));
    Compute(ctx, accumulator);
    accumulator->Unref();
  }

 protected:
  virtual void Compute(OpKernelContext* ctx,
                       ConditionalAccumulatorBase* accumulator) = 0;
};

/**
 * Defines a AccumulateGradientOp, the execution of which adds a gradient to the
 * given ConditionalAccumulator.
 */
class ConditionalAccumulatorBaseApplyGradientOp
    : public ConditionalAccumulatorBaseSyncOpKernel {
 public:
  explicit ConditionalAccumulatorBaseApplyGradientOp(
      OpKernelConstruction* context)
      : ConditionalAccumulatorBaseSyncOpKernel(context) {}

 protected:
  virtual void CheckSignature(OpKernelContext* ctx,
                              ConditionalAccumulatorBase* accumulator) = 0;

  void Compute(OpKernelContext* ctx,
               ConditionalAccumulatorBase* accumulator) override {
    // Check input signature
    CheckSignature(ctx, accumulator);

    // Get input local_step
    const Tensor* local_step_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("local_step", &local_step_tensor));
    if (!TensorShapeUtils::IsScalar(local_step_tensor->shape())) {
      ctx->CtxFailureWithWarning(errors::InvalidArgument(
          "Argument local_step must be scalar, but had bad shape ",
          local_step_tensor->shape().DebugString()));
    }

    // Actually try to apply gradient now
    accumulator->TryApplyGrad(local_step_tensor->scalar<int64>()(), ctx);
  }
};

/**
 * Defines a TakeAccumulatedGradientOp, the execution of which adds a gradient
 * to the given ConditionalAccumulator.
 */
class ConditionalAccumulatorBaseTakeGradientOp
    : public ConditionalAccumulatorBaseAsyncOpKernel {
 public:
  explicit ConditionalAccumulatorBaseTakeGradientOp(
      OpKernelConstruction* context)
      : ConditionalAccumulatorBaseAsyncOpKernel(context) {}

 protected:
  virtual void CheckSignature(OpKernelContext* ctx,
                              ConditionalAccumulatorBase* accumulator,
                              DoneCallback callback) = 0;

  void ComputeAsync(OpKernelContext* ctx,
                    ConditionalAccumulatorBase* accumulator,
                    DoneCallback callback) override {
    // Check signature
    CheckSignature(ctx, accumulator, callback);

    // Get input num_required
    const Tensor* num_required_tensor;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("num_required", &num_required_tensor),
                         callback);
    if (!TensorShapeUtils::IsScalar(num_required_tensor->shape())) {
      ctx->CtxFailureWithWarning(errors::InvalidArgument(
          "Argument num_required must be scalar, but had bad shape ",
          num_required_tensor->shape().DebugString()));
      callback();
    }

    // Actually try to take gradient now
    accumulator->TryTakeGrad(num_required_tensor->scalar<int32>()(), ctx,
                             callback);
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_OP_H_
