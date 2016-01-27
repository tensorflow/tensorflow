/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class T>
class CountUpToOp : public OpKernel {
 public:
  explicit CountUpToOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("limit", &limit_));
  }

  void Compute(OpKernelContext* context) override {
    T before_increment;
    {
      mutex_lock l(*context->input_ref_mutex(0));
      Tensor tensor = context->mutable_input(0, true);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(tensor.shape()),
                  errors::InvalidArgument("input is not a scalar: ",
                                          tensor.shape().DebugString()));
      T* ptr = &tensor.scalar<T>()();
      before_increment = *ptr;
      if (*ptr >= limit_) {
        context->SetStatus(errors::OutOfRange("Reached limit of ", limit_));
        return;
      }
      ++*ptr;
    }
    // Output if no error.
    Tensor* out_tensor;
    OP_REQUIRES_OK(context, context->allocate_output("output", TensorShape({}),
                                                     &out_tensor));
    out_tensor->scalar<T>()() = before_increment;
  }

 private:
  T limit_;
};

#define REGISTER(TYPE)                                                \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("CountUpTo").TypeConstraint<TYPE>("T").Device(DEVICE_CPU), \
      CountUpToOp<TYPE>)

REGISTER(int32);
REGISTER(int64);

#undef REGISTER

}  // namespace tensorflow
