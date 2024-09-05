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
      return absl::OkStatus();
    };
  }

  Status CheckSignature(OpKernelContext* ctx) override {
    TF_RETURN_IF_ERROR(ctx->MatchSignature({}, {DT_STRING_REF}));
    return absl::OkStatus();
  }

  void SetHandleToOutput(OpKernelContext* ctx)
      TF_SHARED_LOCKS_REQUIRED(mu_) override {
    ctx->set_output_ref(0, &mu_, &accumulator_);
  }

  ConditionalAccumulatorOp(const ConditionalAccumulatorOp&) = delete;
  void operator=(const ConditionalAccumulatorOp&) = delete;
};

#define REGISTER_KERNELS(type, dev)                           \
  REGISTER_KERNEL_BUILDER(Name("ConditionalAccumulator")      \
                              .Device(DEVICE_##dev)           \
                              .TypeConstraint<type>("dtype"), \
                          ConditionalAccumulatorOp<dev##Device, type>)

// Resource conditional accumulator
template <typename Device, typename T>
class ResourceConditionalAccumulatorOp : public ConditionalAccumulatorBaseOp {
 public:
  explicit ResourceConditionalAccumulatorOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseOp(context) {}

 protected:
  Creator GetCreator() const override {
    return [this](ConditionalAccumulatorBase** ret) {
      ConditionalAccumulator<Device, T>* accumulator =
          new ConditionalAccumulator<Device, T>(dtype_, shape_, cinfo_.name(),
                                                reduction_type_);
      *ret = accumulator;
      return absl::OkStatus();
    };
  }

  Status CheckSignature(OpKernelContext* ctx) override {
    TF_RETURN_IF_ERROR(ctx->MatchSignature({}, {DT_RESOURCE}));
    return absl::OkStatus();
  }

  void SetHandleToOutput(OpKernelContext* ctx)
      TF_SHARED_LOCKS_REQUIRED(mu_) override {
    auto h = accumulator_.template flat<tstring>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            TypeIndex::Make<ConditionalAccumulatorBase>()));
  }

  ResourceConditionalAccumulatorOp(const ResourceConditionalAccumulatorOp&) =
      delete;
  void operator=(const ResourceConditionalAccumulatorOp&) = delete;
};

#define REGISTER_RESOURCE_KERNELS(type, dev)                     \
  REGISTER_KERNEL_BUILDER(Name("ResourceConditionalAccumulator") \
                              .Device(DEVICE_##dev)              \
                              .TypeConstraint<type>("dtype"),    \
                          ResourceConditionalAccumulatorOp<dev##Device, type>)

// End of Resource conditional accumulator

#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS(type, CPU)

TF_CALL_half(REGISTER_KERNELS_CPU);
TF_CALL_float(REGISTER_KERNELS_CPU);
TF_CALL_double(REGISTER_KERNELS_CPU);

#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS

#define REGISTER_RESOURCE_KERNELS_CPU(type) REGISTER_RESOURCE_KERNELS(type, CPU)

TF_CALL_half(REGISTER_RESOURCE_KERNELS_CPU);
TF_CALL_float(REGISTER_RESOURCE_KERNELS_CPU);
TF_CALL_double(REGISTER_RESOURCE_KERNELS_CPU);

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

  DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) override {
    DataTypeVector expected_inputs;
    expected_inputs = {DT_STRING_REF, DT_INT64};
    expected_inputs.push_back(accumulator->dtype());
    return expected_inputs;
  }

 private:
  AccumulatorApplyGradientOp(const AccumulatorApplyGradientOp&) = delete;
  void operator=(const AccumulatorApplyGradientOp&) = delete;
};

REGISTER_KERNEL_BUILDER(Name("AccumulatorApplyGradient").Device(DEVICE_CPU),
                        AccumulatorApplyGradientOp);

class ResourceAccumulatorApplyGradientOp
    : public ConditionalAccumulatorBaseApplyGradientOp {
 public:
  explicit ResourceAccumulatorApplyGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseApplyGradientOp(context) {}

  DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) override {
    DataTypeVector expected_inputs;
    expected_inputs = {DT_RESOURCE, DT_INT64};
    expected_inputs.push_back(accumulator->dtype());
    return expected_inputs;
  }

 private:
  ResourceAccumulatorApplyGradientOp(
      const ResourceAccumulatorApplyGradientOp&) = delete;
  void operator=(const ResourceAccumulatorApplyGradientOp&) = delete;
};

REGISTER_KERNEL_BUILDER(
    Name("ResourceAccumulatorApplyGradient").Device(DEVICE_CPU),
    ResourceAccumulatorApplyGradientOp);

/**
 * Defines a ConditionalAccumulatorBaseTakeGradientOp, the execution of which
 * returns the average gradient accumulated by the given ConditionalAccumulator.
 */
class AccumulatorTakeGradientOp
    : public ConditionalAccumulatorBaseTakeGradientOp {
 public:
  explicit AccumulatorTakeGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseTakeGradientOp(context) {}

  DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) override {
    return {DT_STRING_REF, DT_INT32};
  }

 private:
  AccumulatorTakeGradientOp(const AccumulatorTakeGradientOp&) = delete;
  void operator=(const AccumulatorTakeGradientOp&) = delete;
};
REGISTER_KERNEL_BUILDER(Name("AccumulatorTakeGradient").Device(DEVICE_CPU),
                        AccumulatorTakeGradientOp);

class ResourceAccumulatorTakeGradientOp
    : public ConditionalAccumulatorBaseTakeGradientOp {
 public:
  explicit ResourceAccumulatorTakeGradientOp(OpKernelConstruction* context)
      : ConditionalAccumulatorBaseTakeGradientOp(context) {}

  DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) override {
    return {DT_RESOURCE, DT_INT32};
  }

 private:
  ResourceAccumulatorTakeGradientOp(const ResourceAccumulatorTakeGradientOp&) =
      delete;
  void operator=(const ResourceAccumulatorTakeGradientOp&) = delete;
};

REGISTER_KERNEL_BUILDER(
    Name("ResourceAccumulatorTakeGradient").Device(DEVICE_CPU),
    ResourceAccumulatorTakeGradientOp);

}  // namespace tensorflow
