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
#include "tensorflow/core/kernels/sparse_conditional_accumulator.h"

namespace tensorflow {

/**
 * Defines a SparseConditionalAccumulatorOp, which constructs a
 * SparseConditionalAccumulator and returns its handle.
 */
template <typename Device, typename T>
class SparseConditionalAccumulatorOp : public ConditionalAccumulatorBaseOp {
 public:
  explicit SparseConditionalAccumulatorOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseOp(context) {}

 protected:
  Creator GetCreator() const override {
    return [this](ConditionalAccumulatorBase** ret) {
      SparseConditionalAccumulator<Device, T>* accumulator =
          new SparseConditionalAccumulator<Device, T>(
              dtype_, shape_, cinfo_.name(), reduction_type_);
      *ret = accumulator;
      return absl::OkStatus();
    };
  }

  // TODO(tanzheny): actually switch it to resource. You won't be able to use
  // it with cond2 otherwise.
  absl::Status CheckSignature(OpKernelContext* ctx) override {
    TF_RETURN_IF_ERROR(ctx->MatchSignature({}, {DT_STRING_REF}));
    return absl::OkStatus();
  }

  void SetHandleToOutput(OpKernelContext* ctx)
      TF_SHARED_LOCKS_REQUIRED(mu_) override {
    ctx->set_output_ref(0, &mu_, &accumulator_);
  }

  SparseConditionalAccumulatorOp(const SparseConditionalAccumulatorOp&) =
      delete;
  void operator=(const SparseConditionalAccumulatorOp&) = delete;
};

#define REGISTER_KERNELS(type, dev)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseConditionalAccumulator") \
                              .Device(DEVICE_##dev)            \
                              .TypeConstraint<type>("dtype"),  \
                          SparseConditionalAccumulatorOp<dev##Device, type>)

#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS(type, CPU)

TF_CALL_half(REGISTER_KERNELS_CPU);
TF_CALL_float(REGISTER_KERNELS_CPU);
TF_CALL_double(REGISTER_KERNELS_CPU);

#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS

/**
 * Defines a SparseAccumulateGradientOp, the execution of which adds a gradient
 * to the given SparseConditionalAccumulator.
 */
class SparseAccumulatorApplyGradientOp
    : public ConditionalAccumulatorBaseApplyGradientOp {
 public:
  explicit SparseAccumulatorApplyGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseApplyGradientOp(context) {}

 protected:
  DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) override {
    DataTypeVector expected_inputs = {DT_STRING_REF, DT_INT64, DT_INT64};
    expected_inputs.push_back(accumulator->dtype());
    expected_inputs.push_back(DT_INT64);
    return expected_inputs;
  }

 private:
  SparseAccumulatorApplyGradientOp(const SparseAccumulatorApplyGradientOp&) =
      delete;
  void operator=(const SparseAccumulatorApplyGradientOp&) = delete;
};

REGISTER_KERNEL_BUILDER(
    Name("SparseAccumulatorApplyGradient").Device(DEVICE_CPU),
    SparseAccumulatorApplyGradientOp);

/**
 * Defines a SparseAccumulatorTakeGradientOp, the execution of which returns the
 * average sparse gradient accumulated by the given ConditionalAccumulator.
 */
class SparseAccumulatorTakeGradientOp
    : public ConditionalAccumulatorBaseTakeGradientOp {
 public:
  explicit SparseAccumulatorTakeGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseTakeGradientOp(context) {}

 protected:
  void CheckSignature(OpKernelContext* ctx,
                      ConditionalAccumulatorBase* accumulator,
                      DoneCallback callback) override {
    // Check signature
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->MatchSignature({DT_STRING_REF, DT_INT32},
                            {DT_INT64, accumulator->dtype(), DT_INT64}),
        callback);
  }

  DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) override {
    return {DT_STRING_REF, DT_INT32};
  }

 private:
  SparseAccumulatorTakeGradientOp(const SparseAccumulatorTakeGradientOp&) =
      delete;
  void operator=(const SparseAccumulatorTakeGradientOp&) = delete;
};

REGISTER_KERNEL_BUILDER(
    Name("SparseAccumulatorTakeGradient").Device(DEVICE_CPU),
    SparseAccumulatorTakeGradientOp);

}  // namespace tensorflow
