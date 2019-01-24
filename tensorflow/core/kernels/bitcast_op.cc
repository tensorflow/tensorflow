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

// See docs in ../ops/array_ops.cc.
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

class BitcastOp : public OpKernel {
 public:
  explicit BitcastOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("T", &input_data_type_));
    OP_REQUIRES_OK(context, context->GetAttr("type", &output_data_type_));
    in_size_ = DataTypeSize(input_data_type_);
    out_size_ = DataTypeSize(output_data_type_);
    int check_size =
        std::max(in_size_, out_size_) % std::min(in_size_, out_size_);
    OP_REQUIRES(
        context, check_size == 0,
        errors::InvalidArgument("cannot convert between datatype ",
                                input_data_type_, " and ", output_data_type_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);

    TensorShape adjusted_shape = input_tensor.shape();
    OP_REQUIRES(context,
                in_size_ >= out_size_ ||
                    (input_tensor.dims() > 0 &&
                     input_tensor.dim_size(input_tensor.dims() - 1) ==
                         out_size_ / in_size_) ||
                    input_tensor.dim_size(input_tensor.dims()) == -1,
                errors::InvalidArgument(
                    "Cannot bitcast from ", DataTypeString(input_data_type_),
                    " to ", DataTypeString(output_data_type_), ": shape ",
                    input_tensor.shape().DebugString()));

    if (out_size_ < in_size_) {
      adjusted_shape.AddDim(in_size_ / out_size_);
    } else if (out_size_ > in_size_) {
      adjusted_shape.RemoveDim(input_tensor.dims() - 1);
    }
    Tensor output_tensor;

    OP_REQUIRES_OK(context,
                   output_tensor.BitcastFrom(input_tensor, output_data_type_,
                                             adjusted_shape));
    context->set_output(0, output_tensor);
  }

  bool IsExpensive() override { return false; }

 private:
  DataType input_data_type_;
  DataType output_data_type_;
  int in_size_;
  int out_size_;
};

REGISTER_KERNEL_BUILDER(Name("Bitcast").Device(DEVICE_CPU), BitcastOp);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Bitcast").Device(DEVICE_GPU), BitcastOp);
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
