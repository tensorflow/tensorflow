/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/parse_ops.cc.

#include <errno.h>
#include <string>

#include "fmt/format.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

template <typename InputType>
class NumberToStringOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    // This is not a deep copy of the input tensor; they will share the same
    // underlying storage.
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<InputType>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("string_tensor",
                                            input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    for (int i = 0; i < input_flat.size(); ++i) {
      output_flat(i) = fmt::to_string<InputType>(input_flat(i));
    }
  }
};

// Registers the currently supported input types.
#define REGISTER(type)                                           \
  REGISTER_KERNEL_BUILDER(Name("NumberToString")                 \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("in_type"),  \
                          NumberToStringOp<type>)
REGISTER(float);
REGISTER(double);
REGISTER(int32);
REGISTER(int64);
#undef REGISTER

}  // namespace tensorflow
