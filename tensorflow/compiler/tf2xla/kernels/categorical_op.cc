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

// XLA implementations of Categorical op.

#include "tensorflow/compiler/tf2xla/kernels/random_ops_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

class CategoricalOp : public XlaOpKernel {
 public:
  explicit CategoricalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        is_gpu_(ctx->device_type().type_string() == DEVICE_GPU_XLA_JIT) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // Get the logits
    const xla::XlaOp& logits = ctx->Input(0);
    TensorShape logits_shape = ctx->InputShape(0);
    int64 num_samples;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &num_samples));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(logits_shape),
                errors::InvalidArgument("logits should be a matrix, got shape ",
                                        logits_shape.DebugString()));
    OP_REQUIRES(ctx, num_samples >= 0,
                errors::InvalidArgument(
                    "num_samples should be nonnegative, got ", num_samples));

    for (int i = 0; i < 2; i++) {
      const int64 dim = logits_shape.dim_size(i);
      OP_REQUIRES(
          ctx, static_cast<int>(dim) == dim,
          errors::InvalidArgument("logits.shape = ", logits_shape.DebugString(),
                                  " too large for int"));
    }

    const int64 batch_size = logits_shape.dim_size(0);
    const int64 num_classes = logits_shape.dim_size(1);

    xla::Shape uniform_shape;
    int class_dimension;
    if (num_samples != 1) {
      std::array<int64, 3> uniform_shape_array = {
          {batch_size, num_samples, num_classes}};
      xla::PrimitiveType uniform_xla_type;
      OP_REQUIRES_OK(ctx,
                     DataTypeToPrimitiveType(input_type(0), &uniform_xla_type));
      uniform_shape =
          xla::ShapeUtil::MakeShape(uniform_xla_type, uniform_shape_array);
      class_dimension = 2;
    } else {
      // Have a special case for when we only need one sample, because
      // dimensions may be padded on architectures with tiled memory layouts, so
      // if the num_classes or batch size is large then this can lead to
      // expensive wasted memory.
      std::array<int64, 2> uniform_shape_array = {{batch_size, num_classes}};
      xla::PrimitiveType uniform_xla_type;
      OP_REQUIRES_OK(ctx,
                     DataTypeToPrimitiveType(input_type(0), &uniform_xla_type));
      uniform_shape =
          xla::ShapeUtil::MakeShape(uniform_xla_type, uniform_shape_array);
      class_dimension = 1;
    }
    xla::PrimitiveType type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(input_type(0), &type));
    xla::XlaOp log_uniforms = GetLogUniforms(uniform_shape, type, ctx);

    // Use Gumbel softmax trick to generate categorical samples.
    // See:
    // https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/
    // TODO(b/68769470): Switch to using a cumulative sum approach.
    auto softmax_entries =
        xla::Sub(logits, log_uniforms,
                 /*broadcast_dimensions=*/{0, class_dimension});

    xla::PrimitiveType xla_output_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(output_type(0), &xla_output_type));
    xla::XlaOp argmax;
    if (is_gpu_) {
      argmax = xla::ArgMaxTwoPass(softmax_entries, xla_output_type,
                                  /*axis=*/class_dimension);
    } else {
      argmax = xla::ArgMax(softmax_entries, xla_output_type,
                           /*axis=*/class_dimension);
    }

    if (num_samples == 1) {
      argmax = xla::Reshape(argmax, {batch_size, 1});
    }

    ctx->SetOutput(0, argmax);
  }

  virtual xla::XlaOp GetLogUniforms(xla::Shape uniform_shape,
                                    xla::PrimitiveType type,
                                    XlaOpKernelContext* ctx) {
    xla::XlaBuilder* builder = ctx->builder();
    LOG(WARNING) << "Warning: Using tf.random.categorical with XLA compilation"
                    " will ignore seeds.";
    // We want a number in (0, 1) rather than [0, 1) or (0, 1]:
    // * log(-log(0)) is ∞.
    // * log(-log(1)) is -∞.
    auto uniforms = xla::RngUniform(
        xla::MinPositiveNormalValue(builder, type),
        xla::One(builder, uniform_shape.element_type()), uniform_shape);
    return xla::Log(-xla::Log(uniforms));
  }

 private:
  bool is_gpu_;
  TF_DISALLOW_COPY_AND_ASSIGN(CategoricalOp);
};

// TODO(b/68769717): Rename this sampler to Categorical.
REGISTER_XLA_OP(Name("Multinomial").CompileTimeConstantInput("num_samples"),
                CategoricalOp);

class StatelessCategoricalOp : public CategoricalOp {
 public:
  explicit StatelessCategoricalOp(OpKernelConstruction* ctx)
      : CategoricalOp(ctx),
        device_type_string_(ctx->device_type().type_string()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  xla::XlaOp GetLogUniforms(xla::Shape uniform_shape, xla::PrimitiveType type,
                            XlaOpKernelContext* ctx) override {
    xla::XlaOp seed = ctx->Input(2);

    xla::XlaBuilder* builder = ctx->builder();
    if (uniform_shape.element_type() == xla::BF16) {
      uniform_shape.set_element_type(xla::F32);
    }
    // We want a number in (0, 1) rather than [0, 1) or (0, 1]:
    // * log(-log(0)) is ∞.
    // * log(-log(1)) is -∞.
    xla::XlaOp uniforms = StatelessRngUniform(
        device_type_string_, seed, uniform_shape,
        xla::MinPositiveNormalValue(builder, uniform_shape.element_type()),
        xla::One(builder, uniform_shape.element_type()));
    return xla::ConvertElementType(xla::Log(-xla::Log(uniforms)), type);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape seed_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    CategoricalOp::Compile(ctx);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessCategoricalOp);
};

REGISTER_XLA_OP(Name("StatelessMultinomial")
                    .CompileTimeConstantInput("num_samples")
                    .TypeConstraint("T", {DT_FLOAT, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                StatelessCategoricalOp);

}  // anonymous namespace
}  // namespace tensorflow
