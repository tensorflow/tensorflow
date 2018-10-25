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
#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
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
    xla::XlaOp result = xla::RngUniform(XlaHelpers::Zero(b, dtype),
                                        XlaHelpers::One(b, dtype), xla_shape);

    ctx->SetOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomUniformOp);
};

REGISTER_XLA_OP(Name("RandomUniform").CompileTimeConstantInput("shape"),
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
      return;
    }

    if (input_shape.dims() == 1) {
      // For R1s, shuffle values by sorting instead of the obvious Fisher-Yates
      // algorithm. Fisher-Yates is simple to implement and correct, but not
      // easily parallelizable. For a sufficiently parallel architecture, it is
      // faster to sort many times, than Fisher-Yates shuffle once.

      // Shuffle values by assigning each value a random key and sorting the
      // keys. Keys can collide causing detectable patterns in the shuffled
      // output. Collisions translates into more ascending sub-sequences in the
      // shuffled output than would be expected by chance. To avoid collisions,
      // the number of possible key values must be sufficiently large.

      // How are more than 2^32 keys created? In each loop iteration, the
      // algorithm sorts by random keys. Conceptually, the earlier iterations
      // are sorting on the lower-order bits of larger keys that are never
      // actually assembled.

      // The expected number of collisions is n - d + d(1 - 1/d)^n, where d is
      // the number of possible keys and n is the number of values. If d = n^2,
      // then the limit as n goes to infinity is 1/2. If d = n^3, then the limit
      // as n goes to infinity is zero.

      // This implementation ensures that the key-space is greater than or equal
      // to the cube of the number of values. The risk of collisions can be
      // further reduced by increasing Exponent at the expense of
      // performance.

      // For Exponent = 2, the expected number of collisions per shuffle is
      // maximized at n = floor((2^32-1)^(1/2)) = 65535 where the expectation is
      // about 1/2.

      // For Exponent = 3, the expected number of collisions per shuffle is
      // maximized at n = floor((2^32-1)^(1/3)) = 1625 where the expectation is
      // about 1/3255.

      // For Exponent = 4, the expected number of collisions per shuffle is
      // maximized at n = floor((2^32-1)^(1/4)) = 255 where the expectation is
      // about 1/132622.
      constexpr int Exponent = 3;
      const int rounds = static_cast<int>(
          std::ceil(Exponent * std::log(num_elements) / std::log(kuint32max)));

      const xla::Shape key_shape =
          xla::ShapeUtil::MakeShape(xla::U32, {num_elements});
      xla::XlaOp zero = xla::ConstantR0(builder, 0U);

      // Unfortunately, xla::RngUniform gives values in the half open interval
      // rather than the closed interval, so instead of 2^32 possible keys there
      // are only 2^32 - 1 (kuint32max).
      xla::XlaOp max_value = xla::ConstantR0(builder, kuint32max);

      xla::XlaOp curr = input;
      for (int i = 0; i < rounds; ++i) {
        xla::XlaOp keys = xla::RngUniform(zero, max_value, key_shape);
        xla::XlaOp sorted = xla::Sort(keys, {curr});
        curr = xla::GetTupleElement(sorted, 1);
      }

      ctx->SetOutput(0, curr);
      return;
    }

    // The Fisher-Yates algorithm.

    // Generate the random swaps for the indices.
    auto swaps_shape = xla::ShapeUtil::MakeShape(xla::S32, {n});
    auto swaps =
        xla::RngUniform(xla::ConstantR0<int32>(builder, 0),
                        xla::ConstantR0<int32>(builder, n), swaps_shape);

    // Generate range(n) as the initial value for the indices to be swapped.
    xla::XlaOp indices = xla::Iota(builder, xla::S32, n);

    // Swap the indices at i and swaps[i].
    auto swap_body_fn = [&](xla::XlaOp i,
                            absl::Span<const xla::XlaOp> loop_vars,
                            xla::XlaBuilder* builder)
        -> xla::StatusOr<std::vector<xla::XlaOp>> {
      auto swaps = loop_vars[0];
      auto indices = loop_vars[1];
      i = xla::Reshape(i, {1});
      // temp = indices[i]
      auto temp = xla::DynamicSlice(indices, i, {1});
      // swap_index = swaps[i]
      auto swap_index = xla::DynamicSlice(swaps, i, {1});
      // swap_value = indices[swaps[i]]
      auto swap_value = xla::DynamicSlice(indices, swap_index, {1});
      // indices[i] = indices[swaps[i]]
      indices = xla::DynamicUpdateSlice(indices, swap_value, i);
      // indices[swaps[i]] = temp
      indices = xla::DynamicUpdateSlice(indices, temp, swap_index);
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
    ctx->SetOutput(0, xla::RngUniform(minval, maxval, xla_shape));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomUniformIntOp);
};

REGISTER_XLA_OP(Name("RandomUniformInt").CompileTimeConstantInput("shape"),
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
    xla::XlaOp result = xla::RngNormal(XlaHelpers::Zero(b, dtype),
                                       XlaHelpers::One(b, dtype), xla_shape);

    ctx->SetOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomStandardNormalOp);
};

REGISTER_XLA_OP(Name("RandomStandardNormal").CompileTimeConstantInput("shape"),
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
    auto uniform = xla::RngUniform(min_positive, one, xla_shape);
    ctx->SetOutput(0, TruncatedNormal(uniform));
  }
};

REGISTER_XLA_OP(Name("TruncatedNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", DT_FLOAT),
                TruncatedNormalOp);

}  // namespace
}  // namespace tensorflow
