/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Position/length can be 32 or 64-bit integers
template <typename T>
class SubstrOp : public OpKernel {
  public:
    using OpKernel::OpKernel;

    void Compute(OpKernelContext* context) override {
      // Get inputs
      const Tensor& input_tensor = context->input(0);
      const Tensor& pos_tensor = context->input(1);
      const Tensor& len_tensor = context->input(2);

      // Validate size of tensors
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(pos_tensor.shape()),
                  errors::InvalidArgument("pos must be a scalar, but got: ",
                                          pos_tensor.shape().DebugString()));
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(len_tensor.shape()),
                  errors::InvalidArgument("len must be a scalar, but got: ",
                                          len_tensor.shape().DebugString()));

      auto input = input_tensor.flat<string>();
      size_t pos = pos_tensor.scalar<T>()(0);
      size_t len = len_tensor.scalar<T>()(0);

      // Allocate output
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output("output", input_tensor.shape(),
                                              &output_tensor));
      auto output = output_tensor->flat<string>();

      // Set output to be substrings of input strings
      for (int i = 0; i < input.size(); i++) {
        OP_REQUIRES(context, pos >= 0 && pos < input(i).size(),
                    errors::InvalidArgument("pos ", pos, 
                                            " out of range for string b'", input(i), "'",
                                            " at index ", i));
        output(i) = input(i).substr(pos, len);
      }
    }
};

#define REGISTER_SUBSTR(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Substr")              \
                          .Device(DEVICE_CPU)         \
                          .TypeConstraint<type>("T"), \
                          SubstrOp<type>);
REGISTER_SUBSTR(int32);
REGISTER_SUBSTR(int64);

}  // namespace tensorflow
