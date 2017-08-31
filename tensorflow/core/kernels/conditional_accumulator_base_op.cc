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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conditional_accumulator_base_op.h"

namespace tensorflow {

/**
 * Defines a AccumulatorSetGlobalStepOp, the execution of which sets the
 * global_step variable of the given ConditionalAccumulator.
 */
class AccumulatorSetGlobalStepOp
    : public ConditionalAccumulatorBaseSyncOpKernel {
 public:
  explicit AccumulatorSetGlobalStepOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseSyncOpKernel(context) {}

 protected:
  void Compute(OpKernelContext* ctx,
               ConditionalAccumulatorBase* accumulator) override {
    // Check signature
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({DT_STRING_REF, DT_INT64}, {}));

    // Get input new_global_step
    const Tensor* new_global_step_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("new_global_step", &new_global_step_tensor));
    if (!TensorShapeUtils::IsScalar(new_global_step_tensor->shape())) {
      ctx->CtxFailureWithWarning(errors::InvalidArgument(
          "Argument num_required must be scalar, but had bad shape ",
          new_global_step_tensor->shape().DebugString()));
    }

    Status status =
        accumulator->SetGlobalStep(new_global_step_tensor->scalar<int64>()());
    if (!status.ok()) ctx->CtxFailureWithWarning(status);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AccumulatorSetGlobalStepOp);
};

REGISTER_KERNEL_BUILDER(Name("AccumulatorSetGlobalStep").Device(DEVICE_CPU),
                        AccumulatorSetGlobalStepOp);

/**
 * Defines a AccumulatorNumAccumulatedOp, which returns the number of gradients
 * that have been accumulated in the given ConditionalAccumulator, and emits it
 * as an output tensor.
 */
class AccumulatorNumAccumulatedOp
    : public ConditionalAccumulatorBaseSyncOpKernel {
 public:
  explicit AccumulatorNumAccumulatedOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseSyncOpKernel(context) {}

 protected:
  void Compute(OpKernelContext* ctx,
               ConditionalAccumulatorBase* accumulator) override {
    // Check signature
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({DT_STRING_REF}, {DT_INT32}));
    Tensor* Taccumulator_size = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &Taccumulator_size));
    Taccumulator_size->flat<int32>().setConstant(
        accumulator->num_accumulated());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AccumulatorNumAccumulatedOp);
};

REGISTER_KERNEL_BUILDER(Name("AccumulatorNumAccumulated").Device(DEVICE_CPU),
                        AccumulatorNumAccumulatedOp);

}  // namespace tensorflow
