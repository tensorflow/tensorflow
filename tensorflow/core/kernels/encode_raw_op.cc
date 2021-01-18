/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

template <typename T>
class EncodeRawOp : public OpKernel {
 public:
  explicit EncodeRawOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const auto& input = context->input(0);
    int64 num_elements = input.NumElements();
    int64 batch_size = input.shape().dim_size(0);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("output", TensorShape({batch_size}),
                                                     &output_tensor));
    if (num_elements == 0) {
      return;
    }
    int64 num_elements_per_sample = num_elements / batch_size;
    const T* input_data = input.flat<T>().data();

    auto output = output_tensor->vec<tstring>();
    for (int64 i = 0; i < batch_size; i++) {
      output(i).resize_uninitialized(num_elements_per_sample * sizeof(T));
      memcpy(output(i).data(), input_data, num_elements_per_sample * sizeof(T));
      input_data += num_elements_per_sample;
    }
  }
};

#define REGISTER(type)                                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("EncodeRaw").Device(DEVICE_CPU).TypeConstraint<type>("dtype"),    \
      EncodeRawOp<type>)

REGISTER(Eigen::half);
REGISTER(float);
REGISTER(double);
REGISTER(int32);
REGISTER(uint16);
REGISTER(uint8);
REGISTER(int16);
REGISTER(int8);
REGISTER(int64);
REGISTER(bool);
REGISTER(complex64);
REGISTER(complex128);

#undef REGISTER

}  // namespace tensorflow
