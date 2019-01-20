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

#ifndef TENSORFLOW_CORE_KERNELS_TYPED_CONDITIONAL_ACCUMULATOR_BASE_H_
#define TENSORFLOW_CORE_KERNELS_TYPED_CONDITIONAL_ACCUMULATOR_BASE_H_

#include "tensorflow/core/kernels/conditional_accumulator_base.h"

namespace tensorflow {

/*
 * TypedConditionalAccumulatorBase is a templated companion of
 * ConditionalAccumulatorBase which allows for subclasses to use different
 * types for the input gradients. (See ConditionalAccumulator and
 * SparseConditionalAccumulator.)
 *
 * TypedConditionalAccumulatorBase defines virtual methods and implements
 * methods which depend on the gradient type. These are mainly methods that are
 * used for adding a new gradient to the accumulator.
 */
template <typename GradientTensorType>
class TypedConditionalAccumulatorBase : public ConditionalAccumulatorBase {
 public:
  TypedConditionalAccumulatorBase(const DataType& dtype,
                                  const PartialTensorShape& shape,
                                  const string& name,
                                  const string& reduction_type)
      : ConditionalAccumulatorBase(dtype, shape, name, reduction_type) {}

  /**
   * Attempts to add a gradient to the accumulator. An ApplyGrad attempt is
   * successful (i.e., has its gradient applied) if its local_step >=
   * current_global_step_ at the time the attempt is processed. Otherwise, if
   * local_step < current_global_step_, the stale gradient is silently dropped.
   *
   * local_step: Time-step at which the gradient was computed.
   * grad:       Gradient tensor to be added to the accumulator.
   * ctx:        Context in which the op is executed.
   */
  void TryApplyGrad(int64 local_step, OpKernelContext* ctx) override {
    {
      mutex_lock l(mu_);
      if (local_step >= current_global_step_) {
        GradientTensorType* grad = nullptr;
        bool is_valid = GetAndValidateTensorInputForApplyGrad(ctx, &grad);
        if (is_valid) {
          if (counter_ > 0) {
            AddToAccumGradFunction(ctx, grad);
          } else {
            AllocateAndAssignToAccumGradFunction(ctx, grad);
          }
          counter_++;
        }
        CleanUpGradTensor(grad);
      }
    }
    FlushUnlocked();
  }

 protected:
  // Virtual methods to be implemented by sub-classes for different datatypes.
  // Implements arithmetic operations specific to datatype.
  virtual void AllocateAndAssignToAccumGradFunction(
      OpKernelContext* ctx, GradientTensorType* grad) = 0;

  virtual void AddToAccumGradFunction(OpKernelContext* ctx,
                                      GradientTensorType* grad) = 0;

  // Method for extracting and validating input provided in an OpKernelContext.
  // Returns true if input was successfully retrieved and is valid.
  // Gradient is returned via the GradientTensorType** tensor.
  virtual bool GetAndValidateTensorInputForApplyGrad(
      OpKernelContext* ctx, GradientTensorType** tensor)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  // Method for cleaning up any memory allocated in
  // GetAndValidateTensorInputForApplyGrad
  virtual void CleanUpGradTensor(GradientTensorType* tensor) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TYPED_CONDITIONAL_ACCUMULATOR_BASE_H_
