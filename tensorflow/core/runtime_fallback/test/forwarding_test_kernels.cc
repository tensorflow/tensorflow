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

// Define sample TF OpKernels that can be called directly from TFRT.

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#endif

namespace tensorflow {

// OpKernels that can be directly called from TFRT must be templated to accept
// alternative implementations of OpKernel, OpKernelConstruction, and
// OpKernelContext.
template <class OpKernelT, class OpKernelConstructionT, class OpKernelContextT>
class ScalarAdd : public OpKernelT {
 public:
  explicit ScalarAdd(OpKernelConstructionT* construction)
      : OpKernelT(construction) {}

  void Compute(OpKernelContextT* ctx) override {
    const Tensor& input0 = ctx->input(0);
    const Tensor& input1 = ctx->input(1);

    Tensor output(input0);
    output.scalar<int32>()() =
        input0.scalar<int32>()() + input1.scalar<int32>()();

    ctx->set_output(0, output);
  }
};

#define SCALAR_ADD_PROPERTIES \
  .Input("scalar_0: int32").Input("scalar_1: int32").Output("out: int32")

#if !defined(IS_MOBILE_PLATFORM)
REGISTER_OP("ScalarAdd") SCALAR_ADD_PROPERTIES;

// When calling ScalarAdd from TF, use the standard OpKernel* types.
REGISTER_KERNEL_BUILDER(
    Name("ScalarAdd").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    ScalarAdd<OpKernel, OpKernelConstruction, OpKernelContext>)
#endif

REGISTER_KERNEL_FALLBACK_OP("ScalarAdd") SCALAR_ADD_PROPERTIES;

// When directly calling ScalarAdd from TFRT, use the TFRTOpKernel*
// types.
REGISTER_KERNEL_FALLBACK_KERNEL(
    "ScalarAdd",
    ScalarAdd<TFRTOpKernel, TFRTOpKernelConstruction, TFRTOpKernelContext>);

template <class OpKernelT, class OpKernelConstructionT, class OpKernelContextT>
class FailingKernel : public OpKernelT {
 public:
  explicit FailingKernel(OpKernelConstructionT* construction)
      : OpKernelT(construction) {}

  void Compute(OpKernelContextT* ctx) override {
    ctx->CtxFailure("filename", 999,
                    errors::Internal("TFRT forwarding error!"));
  }
};

REGISTER_KERNEL_FALLBACK_OP("FailingKernel").Output("out: int32");

REGISTER_KERNEL_FALLBACK_KERNEL(
    "FailingKernel",
    FailingKernel<TFRTOpKernel, TFRTOpKernelConstruction, TFRTOpKernelContext>);

template <class OpKernelT, class OpKernelConstructionT, class OpKernelContextT>
class KernelWithBoolAttr : public OpKernelT {
 public:
  explicit KernelWithBoolAttr(OpKernelConstructionT* construction)
      : OpKernelT(construction) {
    Status s = construction->GetAttr("testattr", &attr_);
    if (!s.ok()) {
      construction->CtxFailure(s);
    }
  }

  void Compute(OpKernelContextT* ctx) override {
    Tensor output(DT_BOOL, TensorShape({}));
    output.flat<bool>()(0) = attr_;
    ctx->set_output(0, output);
  }

 private:
  bool attr_;
};

REGISTER_KERNEL_FALLBACK_OP("KernelWithBoolAttr").Output("out: int32");

REGISTER_KERNEL_FALLBACK_KERNEL(
    "KernelWithBoolAttr",
    KernelWithBoolAttr<TFRTOpKernel, TFRTOpKernelConstruction,
                       TFRTOpKernelContext>);

}  // namespace tensorflow
