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

#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/kernels/shape_util.h"
#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

class VarIsInitializedOp : public XlaOpKernel {
 public:
  explicit VarIsInitializedOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    XlaResource* variable;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &variable));
    ctx->SetOutput(
        0, xla::ConstantR0<bool>(ctx->builder(), variable->initialized()));
  }
};
REGISTER_XLA_OP(Name("VarIsInitializedOp"), VarIsInitializedOp);

class VariableShapeOp : public XlaOpKernel {
 public:
  explicit VariableShapeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    DataType variable_dtype;
    TensorShape shape;
    OP_REQUIRES_OK(ctx,
                   ctx->GetVariableTypeAndShape(0, &variable_dtype, &shape));
    Tensor shape_constant(out_dtype_, TensorShape({shape.dims()}));
    OP_REQUIRES_OK(ctx, TensorShapeToConstant(shape, &shape_constant));
    ctx->SetConstantOutput(0, shape_constant);
  }

 private:
  DataType out_dtype_;
};
REGISTER_XLA_OP(Name("VariableShape").IsMetadataOp(), VariableShapeOp);

class ReadVariableOp : public XlaOpKernel {
 public:
  explicit ReadVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp handle;
    OP_REQUIRES_OK(
        ctx, ctx->ReadVariableInput(0, dtype_, /*shape=*/nullptr, &handle));
    ctx->SetOutput(0, handle);
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(Name("ReadVariableOp").CompilationOnly(), ReadVariableOp);

class AssignVariableOp : public XlaOpKernel {
 public:
  explicit AssignVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx,
                   ctx->AssignVariable(0, ctx->input_type(1), ctx->Input(1)));
  }
};
REGISTER_XLA_OP(Name("AssignVariableOp").CompilationOnly(), AssignVariableOp);

class AssignAddVariableOp : public XlaOpKernel {
 public:
  explicit AssignAddVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    DataType type = ctx->input_type(1);
    xla::XlaOp handle;
    OP_REQUIRES_OK(ctx,
                   ctx->ReadVariableInput(0, type, /*shape=*/nullptr, &handle));
    handle = xla::Add(handle, ctx->Input(1));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, handle));
  }
};
REGISTER_XLA_OP(
    Name("AssignAddVariableOp").TypeConstraint("dtype", kNumericTypes),
    AssignAddVariableOp);

class AssignSubVariableOp : public XlaOpKernel {
 public:
  explicit AssignSubVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    DataType type = ctx->input_type(1);
    xla::XlaOp handle;
    OP_REQUIRES_OK(ctx,
                   ctx->ReadVariableInput(0, type, /*shape=*/nullptr, &handle));
    handle = xla::Sub(handle, ctx->Input(1));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, handle));
  }
};
REGISTER_XLA_OP(
    Name("AssignSubVariableOp").TypeConstraint("dtype", kNumericTypes),
    AssignSubVariableOp);

class ResourceGatherOp : public XlaOpKernel {
 public:
  explicit ResourceGatherOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_dims", &batch_dims_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    DataType type = ctx->expected_output_dtype(0);

    TensorShape input_shape;
    xla::XlaOp input;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, type, &input_shape, &input));

    xla::XlaOp gather;
    OP_REQUIRES_OK(ctx, XlaGatherWithBatchDimsOpImpl(ctx, input, input_shape,
                                                     batch_dims_, &gather));
    ctx->SetOutput(0, gather);
  }

 private:
  int32 batch_dims_;
};
REGISTER_XLA_OP(Name("ResourceGather"), ResourceGatherOp);

class ResourceScatterOp : public XlaOpKernel {
 public:
  explicit ResourceScatterOp(
      OpKernelConstruction* context, bool indices_are_vectors,
      std::function<xla::XlaOp(const xla::XlaOp&, const xla::XlaOp&,
                               xla::XlaBuilder*)>
          combiner)
      : XlaOpKernel(context),
        indices_are_vectors_(indices_are_vectors),
        combiner_(std::move(combiner)) {}

  void Compile(XlaOpKernelContext* context) override {
    xla::XlaBuilder* builder = context->builder();

    DataType dtype = context->input_type(2);
    TensorShape var_shape;
    xla::XlaOp var_value;
    OP_REQUIRES_OK(
        context, context->ReadVariableInput(0, dtype, &var_shape, &var_value));

    const xla::XlaOp indices = context->Input(1);
    const xla::XlaOp updates = context->Input(2);

    auto result = XlaScatter(var_value, updates, indices, indices_are_vectors_,
                             combiner_, builder);
    OP_REQUIRES_OK(context, result.status());
    OP_REQUIRES_OK(context,
                   context->AssignVariable(0, dtype, result.ValueOrDie()));
  }

 private:
  const bool indices_are_vectors_;
  const std::function<xla::XlaOp(const xla::XlaOp&, const xla::XlaOp&,
                                 xla::XlaBuilder*)>
      combiner_;
};

class ResourceScatterAddOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterAddOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
    return xla::Add(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterAdd"), ResourceScatterAddOp);

class ResourceScatterSubOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterSubOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
    return xla::Sub(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterSub"), ResourceScatterSubOp);

class ResourceScatterMulOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterMulOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
    return xla::Mul(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterMul"), ResourceScatterMulOp);

class ResourceScatterDivOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterDivOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
    return xla::Div(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterDiv"), ResourceScatterDivOp);

class ResourceScatterMinOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterMinOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
    return xla::Min(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterMin"), ResourceScatterMinOp);

class ResourceScatterMaxOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterMaxOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
    return xla::Max(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterMax"), ResourceScatterMaxOp);

class ResourceScatterUpdateOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterUpdateOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false,
                          /*combiner=*/{}) {}
};
REGISTER_XLA_OP(Name("ResourceScatterUpdate"), ResourceScatterUpdateOp);

class ResourceScatterNdUpdateOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterNdUpdateOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/true,
                          /*combiner=*/{}) {}
};
REGISTER_XLA_OP(Name("ResourceScatterNdUpdate"), ResourceScatterNdUpdateOp);

class ResourceScatterNdAddOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterNdAddOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/true,
                          /*combiner=*/Combine) {}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
    return xla::Add(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterNdAdd"), ResourceScatterNdAddOp);

class ResourceScatterNdSubOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterNdSubOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/true,
                          /*combiner=*/Combine) {}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
    return xla::Sub(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterNdSub"), ResourceScatterNdSubOp);

}  // namespace
}  // namespace tensorflow
