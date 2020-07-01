/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/tensor_map.h"
#include <iostream>
using namespace std;

namespace tensorflow {

class EmptyTensorMap : public OpKernel {
 public:
  explicit EmptyTensorMap(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &element_dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& max_num_elements_t = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(max_num_elements_t.shape()),
        errors::InvalidArgument(
            "max_num_elements expected to be a scalar ",
            "but got shape: ", max_num_elements_t.shape().DebugString()));
    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    g(ctx, ctx->allocate_output(0, TensorShape{}, &result, attr));
    TensorMap empty;
    empty.element_dtype = element_dtype_;
    empty.max_num_elements = max_num_elements_t.scalar<int32>()();
    PartialTensorShape element_shape;
    //OP_REQUIRES_OK(ctx, TensorShapeFromTensor(ctx->input(0), &element_shape));
    empty.element_shape = element_shape;
    result->scalar<Variant>()() = std::move(empty);
  }

 private:
  DataType element_dtype_;
};



class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    cout << "Hello World - Op" << endl;
    // Grab the input tensor
    const Tensor& input_tensor = c->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(c, c->allocate_output(0, input_tensor.shape(), 
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0
    const int N = input.size();
    for (int i=1; i<N; i++) {
      output_flat(i) = 0;
    }
    
    // Preserve the first input value if possible
    if (N > 0) output_flat(0) = input(0);
  }
};



}  // namespace tensorflow
