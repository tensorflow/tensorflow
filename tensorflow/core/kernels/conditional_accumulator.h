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

#ifndef TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_H_
#define TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_H_

#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/typed_conditional_accumulator_base.h"

namespace tensorflow {

/**
 * An aggregation object for adding dense gradients.
 *
 * The two main methods of this class are TryApplyGrad and TryTakeGrad.
 *
 * TryApplyGrad tries add a gradient to the accumulator. The attempt is
 * successful if local_step >= global_step, i.e., if the gradient is not stale,
 * having been computed using up-to-date information. Otherwise, the gradient is
 * silently dropped.
 *
 * TryTakeGrad logs an attempt to read the average gradient. The attempt is
 * blocked until the number of gradients accumulated (via TryApplyGrad) is equal
 * or exceeds the number requested by TryTakeGrad.
 * Once this condition is satisfied, the following actions are taken:
 * (1) the value of the average gradient is returned
 * (2) the count of accumulated gradients is reset to 0
 * (3) the internal global_step value (current_global_step_) is incremented by 1
 *
 * ConditionalAccumulator is the datatype-dependent templated sub-class of
 * ConditionalAccumulatorBase. It implements the virtual arithmetic methods that
 * are used by for aggregating, averaging, allocating, returning dense Tensors.
 */
template <typename Device, typename T>
class ConditionalAccumulator
    : public TypedConditionalAccumulatorBase<const Tensor> {
 public:
  // Args:
  //   dtype: The datatype of the gradients to be accumulated.
  //   shape: The shape of the accumulated gradients.
  //   name:  A name to use for the ConditionalAccumulator.
  //   reduction_type: The reduction type, i.e., MEAN or SUM
  ConditionalAccumulator(const DataType& dtype, const PartialTensorShape& shape,
                         const string& name, const string& reduction_type)
      : TypedConditionalAccumulatorBase<const Tensor>(dtype, shape, name,
                                                      reduction_type) {}
  ~ConditionalAccumulator() override{};

 protected:
  // accum_grad is the tensor that holds the aggregate gradient.
  // It is initialized the first time ApplyGrad is called.
  Tensor* accum_grad_ = nullptr;
  PersistentTensor accum_grad_persistent_;

  functor::SetZeroFunctor<Device, T> set_zero_functor_;

  Status ValidateShape(const Tensor* tensor)
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    // Must be compatible with accumulated gradient if available
    if (counter_ > 0) {
      if (!accum_grad_->shape().IsSameSize(tensor->shape())) {
        return errors::InvalidArgument("Shape mismatch: expected ",
                                       accum_grad_->shape().DebugString(),
                                       ", got ", tensor->shape().DebugString());
      }
    }
    // Must also be compatible with given shape
    if (!shape_.IsCompatibleWith(tensor->shape())) {
      return errors::InvalidArgument("Shape mismatch: expected ",
                                     shape_.DebugString(), ", got ",
                                     tensor->shape().DebugString());
    }
    return Status::OK();
  }

  void AllocateAndAssignToAccumGradFunction(OpKernelContext* ctx,
                                            const Tensor* grad) override {
    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    ctx->allocate_persistent(dtype_, grad->shape(), &accum_grad_persistent_,
                             &accum_grad_)
        .IgnoreError();
    accum_grad_->flat<T>().device(ctx->template eigen_device<Device>()) =
        grad->flat<T>();
  }

  void AddToAccumGradFunction(OpKernelContext* ctx,
                              const Tensor* grad) override {
    accum_grad_->flat<T>().device(ctx->template eigen_device<Device>()) +=
        grad->flat<T>();
  }

  void DivideAccumGradByCounter(OpKernelContext* ctx) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    Tensor c(DataTypeToEnum<T>::value, {});
    c.scalar<T>()() = TypeConverter<T, int>::ConvertUToT(this->counter_);
    this->accum_grad_->template flat<T>().device(
        ctx->template eigen_device<Device>()) =
        this->accum_grad_->template flat<T>() / c.scalar<T>()();
  }

  bool SetOutput(OpKernelContext* ctx) override {
    ctx->set_output(0, *accum_grad_);
    return true;
  }

  bool GetAndValidateTensorInputForApplyGrad(OpKernelContext* ctx,
                                             const Tensor** tensor) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    // Get input gradient tensor
    const Tensor* grad_tensor;
    OP_REQUIRES_OK_BOOLEAN(ctx, ctx->input("gradient", &grad_tensor));
    *tensor = grad_tensor;
    OP_REQUIRES_OK_BOOLEAN(ctx, this->ValidateShape(*tensor));
    return true;
  }

  void CleanUpGradTensor(const Tensor* tensor) override {
    // do nothing
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ConditionalAccumulator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_H_
