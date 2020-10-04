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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

// This OpKernel implements the FakeParam Op for XLA JIT devices. Create zeros
// with the appropriate shape for FakeParam op.
class XlaFakeParamOp : public XlaOpKernel {
 public:
  explicit XlaFakeParamOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    DataType dtype;
    // Tensor shape can be unknown.
    PartialTensorShape tensor_shape;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &tensor_shape));
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, tensor_shape, &shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    ctx->SetOutput(0, xla::Zeros(b, shape_));
  }

 private:
  xla::Shape shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaFakeParamOp);
};

REGISTER_XLA_OP(Name("FakeParam"), XlaFakeParamOp);

}  // namespace tensorflow
