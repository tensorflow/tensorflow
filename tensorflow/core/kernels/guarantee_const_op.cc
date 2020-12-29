/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {
namespace {

// Refer to the Op description for detailed comments.
class GuaranteeConstOp : public OpKernel {
 public:
  explicit GuaranteeConstOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const DataType input_dtype = ctx->input_dtype(0);
    OP_REQUIRES(ctx, input_dtype != DT_RESOURCE,
                errors::InvalidArgument(
                    "Input tensor cannot be a resource variable handle."));
    const Tensor& input_tensor = ctx->input(0);
    Tensor* output = nullptr;
    if (!ctx->forward_input_to_output_with_shape(0, 0, input_tensor.shape(),
                                                 &output)) {
      ctx->set_output(0, input_tensor);
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("GuaranteeConst").Device(DEVICE_CPU),
                        GuaranteeConstOp);

}  // namespace
}  // namespace tensorflow
