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

// XLA implementations of Random ops
// TODO(misard,phawkins): handle random number generator seeds/states correctly.
// TODO(misard,phawkins): add tests.

#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/lib/random.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/lib/while_loop.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class RandomUniformOp : public XlaOpKernel {
 public:
  explicit RandomUniformOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));

    const DataType dtype = output_type(0);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp result = b->RngUniform(XlaHelpers::Zero(b, dtype),
                                      XlaHelpers::One(b, dtype), xla_shape);

    ctx->SetOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomUniformOp);
};

REGISTER_XLA_OP(Name("RandomUniform").CompileTimeConstInput("shape"),
                RandomUniformOp);

class RandomShuffleOp : public XlaOpKernel {
 public:
  explicit RandomShuffleOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    xla::XlaOp input = ctx->Input(0);
    TensorShape input_shape = ctx->InputShape(0);
    const int64 n = input_shape.dim_size(0);
    int64 num_elements = 1;
    for (tensorflow::TensorShapeDim dimension : input_shape) {
      num_elements *= dimension.size;
    }
    if (num_elements <= 1 || n <= 1) {
      // No shuffling is required, so copy input directly to output
      ctx->SetOutput(0, input);
    } else {
      // Generate the random swaps for the indices.
      auto swaps_shape = xla::ShapeUtil::MakeShape(xla::S32, {n});
      auto swaps =
          builder->RngUniform(builder->ConstantR0<int32>(0),
                              builder->ConstantR0<int32>(n), swaps_shape);

      // Generate range(n) as the initial value for the indices to be swapped.
      xla::XlaOp indices;
      TF_CHECK_OK(XlaHelpers::Iota(builder, DataType::DT_INT32, n, &indices));

      // Swap the indices at i and swaps[i].
      auto swap_body_fn = [&](xla::XlaOp i,
                              gtl::ArraySlice<xla::XlaOp> loop_vars,
                              xla::XlaBuilder* builder)
          -> xla::StatusOr<std::vector<xla::XlaOp>> {
        auto swaps = loop_vars[0];
        auto indices = loop_vars[1];
        i = builder->Reshape(i, {1});
        // temp = indices[i]
        auto temp = builder->DynamicSlice(indices, i, {1});
        // swap_index = swaps[i]
        auto swap_index = builder->DynamicSlice(swaps, i, {1});
        // swap_value = indices[swaps[i]]
        auto swap_value = builder->DynamicSlice(indices, swap_index, {1});
        // indices[i] = indices[swaps[i]]
        indices = builder->DynamicUpdateSlice(indices, swap_value, i);
        // indices[swaps[i]] = temp
        indices = builder->DynamicUpdateSlice(indices, temp, swap_index);
        return std::vector<xla::XlaOp>{swaps, indices};
      };
      // for i in range(n):
      auto swap_loop_result =
          XlaForEachIndex(n, xla::S32, swap_body_fn, {swaps, indices},
                          "indices_swap_loop", builder)
              .ValueOrDie();
      auto swapped_indices = swap_loop_result[1];

      // Gather the data using the swapped indices as the shuffled order.
      auto indices_tensor_shape = TensorShape({n});
      DataType type = ctx->expected_output_dtype(0);
      xla::XlaOp gather;
      OP_REQUIRES_OK(ctx, XlaGather(input, input_shape, swapped_indices,
                                    indices_tensor_shape,
                                    /*axis=*/0, /*indices_are_nd=*/false, type,
                                    DT_INT32, builder, &gather));
      ctx->SetOutput(0, gather);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomShuffleOp);
};

REGISTER_XLA_OP(Name("RandomShuffle"), RandomShuffleOp);

class RandomUniformIntOp : public XlaOpKernel {
 public:
  explicit RandomUniformIntOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeToXLAShape(input_type(1), shape, &xla_shape));

    const TensorShape minval_shape = ctx->InputShape(1);
    const TensorShape maxval_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval_shape),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval_shape),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval_shape.DebugString()));

    auto minval = ctx->Input(1);
    auto maxval = ctx->Input(2);
    ctx->SetOutput(0, ctx->builder()->RngUniform(minval, maxval, xla_shape));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomUniformIntOp);
};

REGISTER_XLA_OP(Name("RandomUniformInt").CompileTimeConstInput("shape"),
                RandomUniformIntOp);

class RandomStandardNormalOp : public XlaOpKernel {
 public:
  explicit RandomStandardNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaBuilder* b = ctx->builder();

    // Normal distribution with a mean of 0 and a standard deviation of 1:
    xla::XlaOp result = b->RngNormal(XlaHelpers::Zero(b, dtype),
                                     XlaHelpers::One(b, dtype), xla_shape);

    ctx->SetOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomStandardNormalOp);
};

REGISTER_XLA_OP(Name("RandomStandardNormal").CompileTimeConstInput("shape"),
                RandomStandardNormalOp);

class TruncatedNormalOp : public XlaOpKernel {
 public:
  explicit TruncatedNormalOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaBuilder* b = ctx->builder();

    xla::XlaOp one = XlaHelpers::FloatLiteral(b, dtype, 1.0);
    xla::XlaOp min_positive =
        XlaHelpers::FloatLiteral(b, dtype, std::numeric_limits<float>::min());
    auto uniform = b->RngUniform(min_positive, one, xla_shape);
    auto truncated_normal_or_status = TruncatedNormal(dtype, uniform, b);
    OP_REQUIRES_OK(ctx, truncated_normal_or_status.status());
    ctx->SetOutput(0, truncated_normal_or_status.ValueOrDie());
  }
};

REGISTER_XLA_OP(Name("TruncatedNormal")
                    .CompileTimeConstInput("shape")
                    .TypeConstraint("dtype", DT_FLOAT),
                TruncatedNormalOp);

}  // anonymous namespace
}  // namespace tensorflow
