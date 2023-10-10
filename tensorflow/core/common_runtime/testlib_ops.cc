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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace test {

// ErrorOp::Compute returns an error.
REGISTER_OP("Error")
    .Input("in: T")
    .Output("out: T")
    .Attr("T: type")
    .Attr("message: string")
    .Attr("log_error: bool = false")
    .SetShapeFn(shape_inference::UnknownShape);
class ErrorOp : public OpKernel {
 public:
  explicit ErrorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("message", &errmsg_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("log_error", &log_error_));
  }

  void Compute(OpKernelContext* ctx) override {
    // Log only when CancellationManager is set to skip logging when Compute()
    // is called during the optimization phase.
    if (ctx->cancellation_manager() && log_error_) {
      LOG(ERROR) << "ErrorOp: " << errmsg_;
    }
    ctx->SetStatus(errors::Internal(errmsg_));
  }

 private:
  string errmsg_;
  bool log_error_ = false;
};
REGISTER_KERNEL_BUILDER(Name("Error").Device(DEVICE_CPU), ErrorOp);

REGISTER_OP("InvalidRefType")
    .Output("out: Ref(TIn)")
    .Attr("TIn: type")
    .Attr("TOut: type")
    .SetShapeFn(shape_inference::UnknownShape);
class InvalidRefType : public OpKernel {
 public:
  explicit InvalidRefType(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("TOut", &dtout_));
    output_ = Tensor(dtout_, TensorShape({}));
  }

  void Compute(OpKernelContext* ctx) override {
    ctx->set_output_ref(0, &mu_, &output_);
  }

 private:
  DataType dtout_;
  mutex mu_;
  Tensor output_;
};
REGISTER_KERNEL_BUILDER(Name("InvalidRefType").Device(DEVICE_CPU),
                        InvalidRefType);

// DelayOp::AsyncCompute sleeps for "micros"-econd and then returns
// its input.
REGISTER_OP("Delay")
    .Input("in: T")
    .Output("out: T")
    .Attr("T: type")
    .Attr("micros: int")
    .SetShapeFn(shape_inference::UnchangedShape);
class DelayOp : public AsyncOpKernel {
 public:
  explicit DelayOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("micros", &micros_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    ctx->set_output(0, ctx->input(0));
    ctx->env()->SchedClosureAfter(micros_, done);
  }

 private:
  int64_t micros_;
};
REGISTER_KERNEL_BUILDER(Name("Delay").Device(DEVICE_CPU), DelayOp);

}  // namespace test
}  // namespace tensorflow
