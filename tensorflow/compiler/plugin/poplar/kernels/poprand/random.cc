/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/stream_executor_util.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"

#include "absl/container/flat_hash_set.h"

using namespace xla::poplarplugin;

namespace tensorflow {

namespace {

class PopopsTruncatedNormalOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopopsTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  ~PopopsTruncatedNormalOp() override{};
  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaBuilder* b = ctx->builder();

    xla::XlaOp output =
        xla::CustomCall(b,
                        GetPoplibsCustomOpTargetString(
                            PoplibsOp::Poprand, PoplibsOp::TruncatedNormal),
                        {}, xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsTruncatedNormalOp);
};

class PopopsStatelessOp : public XlaOpKernel, public IpuOpKernel {
 public:
  explicit PopopsStatelessOp(OpKernelConstruction* ctx, PoplibsOp::Op op_type)
      : XlaOpKernel(ctx), op_type_(op_type) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));

    xla::XlaOp output = xla::CustomCall(
        ctx->builder(),
        GetPoplibsCustomOpTargetString(PoplibsOp::Poprand, op_type_),
        {ctx->Input(1)}, xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  PoplibsOp::Op op_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(PopopsStatelessOp);
};

class PopopsStatelessRandomUniformOp : public PopopsStatelessOp {
 public:
  explicit PopopsStatelessRandomUniformOp(OpKernelConstruction* ctx)
      : PopopsStatelessOp(ctx, PoplibsOp::StatelessRandomUniform) {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsStatelessRandomUniformOp);
};

class PopopsStatelessRandomNormalOp : public PopopsStatelessOp {
 public:
  explicit PopopsStatelessRandomNormalOp(OpKernelConstruction* ctx)
      : PopopsStatelessOp(ctx, PoplibsOp::StatelessRandomNormal) {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsStatelessRandomNormalOp);
};

class PopopsStatelessTruncatedNormalOp : public PopopsStatelessOp {
 public:
  explicit PopopsStatelessTruncatedNormalOp(OpKernelConstruction* ctx)
      : PopopsStatelessOp(ctx, PoplibsOp::StatelessTruncatedNormal) {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsStatelessTruncatedNormalOp);
};

class PopopsStatelessRandomUniformIntOp : public XlaOpKernel,
                                          public IpuOpKernel {
 public:
  explicit PopopsStatelessRandomUniformIntOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    TensorShape minval_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval_shape),
                errors::InvalidArgument("minval must be scalar, got shape ",
                                        minval_shape.DebugString()));
    TensorShape maxval_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval_shape),
                errors::InvalidArgument("minval must be scalar, got shape ",
                                        maxval_shape.DebugString()));

    xla::XlaOp output = xla::CustomCall(
        ctx->builder(),
        GetPoplibsCustomOpTargetString(PoplibsOp::Poprand,
                                       PoplibsOp::StatelessRandomUniformInt),
        {ctx->Input(1), ctx->Input(2), ctx->Input(3)}, xla_shape,
        attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsStatelessRandomUniformIntOp);
};

REGISTER_XLA_OP(Name("TruncatedNormal")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF}),
                PopopsTruncatedNormalOp);

REGISTER_XLA_OP(Name("StatelessTruncatedNormal")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF})
                    .TypeConstraint("Tseed", DT_INT32),
                PopopsStatelessTruncatedNormalOp);

REGISTER_XLA_OP(Name("StatelessRandomUniform")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF})
                    .TypeConstraint("Tseed", DT_INT32),
                PopopsStatelessRandomUniformOp);

REGISTER_XLA_OP(Name("StatelessRandomUniformInt")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_INT32})
                    .TypeConstraint("Tseed", DT_INT32),
                PopopsStatelessRandomUniformIntOp);

REGISTER_XLA_OP(Name("StatelessRandomNormal")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF})
                    .TypeConstraint("Tseed", DT_INT32),
                PopopsStatelessRandomNormalOp);

}  // namespace

}  // namespace tensorflow
