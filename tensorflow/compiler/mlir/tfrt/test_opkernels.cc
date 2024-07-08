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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace tf_mlrt {

REGISTER_OP("TestAsyncIdentity")
    .Input("in: T")
    .Output("out: T")
    .Attr(
        "T: {bfloat16, half, float, double, uint8, int8, int16, uint32, int32, "
        "int64, complex64, complex128}")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

class TestAsyncIdentityKernel : public AsyncOpKernel {
 public:
  explicit TestAsyncIdentityKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& in = ctx->input(0);
    ctx->set_output(0, in);
    done();
  }

 private:
  TestAsyncIdentityKernel(const TestAsyncIdentityKernel&) = delete;
  void operator=(const TestAsyncIdentityKernel&) = delete;
};

REGISTER_KERNEL_BUILDER(Name("TestAsyncIdentity").Device(DEVICE_CPU),
                        TestAsyncIdentityKernel);

}  // namespace tf_mlrt
}  // namespace tensorflow
