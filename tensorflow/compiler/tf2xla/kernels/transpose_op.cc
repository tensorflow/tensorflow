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

// XLA-specific Transpose Op. This is very different to the Eigen
// version in third_party/tensorflow because XLA's reshape neatly
// handles all transposes, while Eigen needs a restricted DoTranspose
// helper.

#include "tensorflow/core/kernels/transpose_op.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {
namespace {

class TransposeOp : public XlaOpKernel {
 public:
  explicit TransposeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape perm_tensor_shape = ctx->InputShape(1);

    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm_tensor_shape),
                errors::InvalidArgument("perm must be a vector, not ",
                                        perm_tensor_shape.DebugString()));

    const int dims = input_shape.dims();
    OP_REQUIRES(ctx, dims == perm_tensor_shape.num_elements(),
                errors::InvalidArgument("transpose expects a vector of size ",
                                        input_shape.dims(),
                                        ". But input(1) is a vector of size ",
                                        perm_tensor_shape.num_elements()));

    xla::Literal literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputReshaped(1, {dims}, &literal));

    std::vector<int32> perm(dims);
    std::copy(literal.s32s().begin(), literal.s32s().end(), perm.begin());

    std::vector<int64> transposed_order;
    // Check whether permutation is a permutation of integers of [0 .. dims).
    gtl::InlinedVector<bool, 8> bits(dims);
    bool is_identity = true;
    for (int i = 0; i < dims; ++i) {
      const int32 d = perm[i];
      OP_REQUIRES(
          ctx, 0 <= d && d < dims,
          errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
      bits[d] = true;
      transposed_order.push_back(d);
      if (d != i) {
        is_identity = false;
      }
    }
    for (int i = 0; i < dims; ++i) {
      OP_REQUIRES(
          ctx, bits[i],
          errors::InvalidArgument(i, " is missing from 'perm' argument."));
    }

    // 0-D, 1-D, and identity transposes do nothing.
    if (dims <= 1 || is_identity) {
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }

    ctx->SetOutput(0,
                   ctx->builder()->Transpose(ctx->Input(0), transposed_order));
  }
};

REGISTER_XLA_OP(Name("Transpose").CompileTimeConstInput("perm"), TransposeOp);

// InvertPermutation frequently forms part of the gradient of Transpose.
//
// inv = InvertPermutationOp(T<int32> p) takes a permutation of
// integers 0, 1, ..., n - 1 and returns the inverted
// permutation of p. I.e., inv[p[i]] == i, for i in [0 .. n).
//
// REQUIRES: input is a vector of int32.
// REQUIRES: input is a permutation of 0, 1, ..., n-1.

class InvertPermutationOp : public XlaOpKernel {
 public:
  explicit InvertPermutationOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx,
                FastBoundsCheck(ctx->InputShape(0).num_elements(),
                                std::numeric_limits<int32>::max()),
                errors::InvalidArgument("permutation of nonnegative int32s "
                                        "must have <= int32 max elements"));

    std::vector<int64> perm;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(0, &perm));

    int size = perm.size();

    std::vector<int32> output(size);
    std::fill_n(output.data(), size, -1);
    for (int i = 0; i < size; ++i) {
      const int64 d = perm[i];
      OP_REQUIRES(ctx, FastBoundsCheck(d, size),
                  errors::InvalidArgument(d, " is not between 0 and ", size));
      OP_REQUIRES(ctx, output[d] == -1,
                  errors::InvalidArgument(d, " is duplicated in the input."));
      output[d] = i;
    }

    ctx->SetOutput(0, ctx->builder()->ConstantR1<int32>(output));
  }
};

REGISTER_XLA_OP(Name("InvertPermutation")
                    .TypeConstraint("T", DT_INT32)
                    .CompileTimeConstInput("x"),
                InvertPermutationOp);

}  // namespace
}  // namespace tensorflow
