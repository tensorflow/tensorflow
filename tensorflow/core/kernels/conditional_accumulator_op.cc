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

#include "tensorflow/core/kernels/conditional_accumulator.h"
#include "tensorflow/core/kernels/conditional_accumulator_base_op.h"

namespace tensorflow {

/**
 * Defines a ConditionalAccumulatorOp, which constructs a ConditionalAccumulator
 * and returns its handle.
 */
template <typename Device, typename T>
class ConditionalAccumulatorOp : public ConditionalAccumulatorBaseOp {
 public:
  explicit ConditionalAccumulatorOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseOp(context) {}

 protected:
  Creator GetCreator() const override {
    return [this](ConditionalAccumulatorBase** ret) {
      ConditionalAccumulator<Device, T>* accumulator =
          new ConditionalAccumulator<Device, T>(dtype_, shape_, cinfo_.name(),
                                                reduction_type_);
      *ret = accumulator;
      return Status::OK();
    };
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ConditionalAccumulatorOp);
};

#define REGISTER_KERNELS(type, dev)                           \
  REGISTER_KERNEL_BUILDER(Name("ConditionalAccumulator")      \
                              .Device(DEVICE_##dev)           \
                              .TypeConstraint<type>("dtype"), \
                          ConditionalAccumulatorOp<dev##Device, type>)

#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS(type, CPU)

TF_CALL_half(REGISTER_KERNELS_CPU);
TF_CALL_float(REGISTER_KERNELS_CPU);
TF_CALL_double(REGISTER_KERNELS_CPU);

#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS

/**
 * Defines a AccumulateGradientOp, the execution of which adds a gradient to the
 * given ConditionalAccumulator.
 */
class AccumulatorApplyGradientOp
    : public ConditionalAccumulatorBaseApplyGradientOp {
 public:
  explicit AccumulatorApplyGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseApplyGradientOp(context) {}

 protected:
  void CheckSignature(OpKernelContext* ctx,
                      ConditionalAccumulatorBase* accumulator) override {
    // Check input signature
    DataTypeVector expected_inputs = {DT_STRING_REF, DT_INT64};
    expected_inputs.push_back(accumulator->dtype());
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AccumulatorApplyGradientOp);
};

REGISTER_KERNEL_BUILDER(Name("AccumulatorApplyGradient").Device(DEVICE_CPU),
                        AccumulatorApplyGradientOp);

/**
 * Defines a ConditionalAccumulatorBaseTakeGradientOp, the execution of which
 * returns the average gradient accumulated by the given ConditionalAccumulator.
 */
class AccumulatorTakeGradientOp
    : public ConditionalAccumulatorBaseTakeGradientOp {
 public:
  explicit AccumulatorTakeGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseTakeGradientOp(context) {}

 protected:
  void CheckSignature(OpKernelContext* ctx,
                      ConditionalAccumulatorBase* accumulator,
                      DoneCallback callback) override {
    // Check signature
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->MatchSignature({DT_STRING_REF, DT_INT32}, {accumulator->dtype()}),
        callback);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AccumulatorTakeGradientOp);
};

REGISTER_KERNEL_BUILDER(Name("AccumulatorTakeGradient").Device(DEVICE_CPU),
                        AccumulatorTakeGradientOp);

}  // namespace tensorflow
