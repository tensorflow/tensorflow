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

#include <vector>

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/xla_builder.h"
#include "xla/primitive_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace {

class TransposeOp : public XlaOpKernel {
 public:
  explicit TransposeOp(OpKernelConstruction* ctx, bool conjugate = false)
      : XlaOpKernel(ctx), conjugate_(conjugate) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape("x");
    const TensorShape perm_tensor_shape = ctx->InputShape("perm");

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

    std::vector<int64_t> perm;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector("perm", &perm));

    std::vector<int64_t> transposed_order;
    // Check whether permutation is a permutation of integers of [0 .. dims).
    absl::InlinedVector<bool, 8> bits(dims);
    bool is_identity = true;
    for (int i = 0; i < dims; ++i) {
      const int64_t d = perm[i];
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

    xla::XlaOp transposed;
    // 0-D, 1-D, and identity transposes do nothing.
    if (dims <= 1 || is_identity) {
      transposed = ctx->Input("x");
    } else {
      transposed = xla::Transpose(ctx->Input("x"), transposed_order);
    }

    // Conjugate the transposed result if this is ConjugateTransposeOp.
    if (conjugate_) {
      ctx->SetOutput(0, xla::Conj(transposed));
    } else {
      ctx->SetOutput(0, transposed);
    }
  }

 private:
  const bool conjugate_;
};

class ConjugateTransposeOp : public TransposeOp {
 public:
  explicit ConjugateTransposeOp(OpKernelConstruction* ctx)
      : TransposeOp(ctx, /*conjugate=*/true) {}
};

REGISTER_XLA_OP(Name("Transpose").CompileTimeConstantInput("perm"),
                TransposeOp);

REGISTER_XLA_OP(Name("ConjugateTranspose").CompileTimeConstantInput("perm"),
                ConjugateTransposeOp);

// InvertPermutation frequently forms part of the gradient of Transpose.
//
// inv = InvertPermutationOp(p) takes a permutation of
// integers 0, 1, ..., n - 1 and returns the inverted
// permutation of p. I.e., inv[p[i]] == i, for i in [0 .. n).
//
// REQUIRES: input is a vector of int32 or int64.
// REQUIRES: input is a permutation of 0, 1, ..., n-1.

class InvertPermutationOp : public XlaOpKernel {
 public:
  explicit InvertPermutationOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    DataType dtype = ctx->expected_output_dtype(0);
    Status status;
    switch (dtype) {
      case DT_INT32:
        InvertPermutation<int32>(ctx);
        break;
      case DT_INT64:
        InvertPermutation<int64_t>(ctx);
        break;
      default:
        // This should never happen since we restrict this kernel to only match
        // inputs with supported Tensor datatype.
        OP_REQUIRES_OK(ctx, errors::InvalidArgument(
                                "InvertPermutation expects x as either ",
                                "int32 or int64, not ", DataTypeString(dtype)));
    }
  }

  template <typename T>
  void InvertPermutation(XlaOpKernelContext* ctx) {
    OP_REQUIRES(ctx,
                FastBoundsCheck(ctx->InputShape(0).num_elements(),
                                std::numeric_limits<T>::max()),
                errors::InvalidArgument(
                    "permutation of nonnegative integers must have <= ",
                    std::numeric_limits<T>::max(), " elements"));

    auto e = ctx->InputExpression(0);
    auto* client = ctx->compiler() ? ctx->compiler()->client() : nullptr;
    auto tensor_or_status = e.ResolveConstant(client);
    OP_REQUIRES_OK(ctx, tensor_or_status.status());
    // If the input is a constant, we also want the output to be a constant.
    // Some models rely on the result of InvertPermutation being a constant.
    // TODO(b/32495713): Remove this when we can check whether Scatter is
    // constant. Right now, we always assume it is non-constant because we don't
    // check the embedded computation.
    if (tensor_or_status.value().has_value()) {
      std::vector<int64_t> perm;
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(0, &perm));

      int size = perm.size();

      std::vector<T> output(size);
      std::fill_n(output.data(), size, -1);
      for (int i = 0; i < size; ++i) {
        const int64_t d = perm[i];
        OP_REQUIRES(ctx, FastBoundsCheck(d, size),
                    errors::InvalidArgument(d, " is not between 0 and ", size));
        OP_REQUIRES(ctx, output[d] == -1,
                    errors::InvalidArgument(d, " is duplicated in the input."));
        output[d] = i;
      }

      ctx->SetOutput(0, xla::ConstantR1<T>(ctx->builder(), output));
    } else {
      auto indices = ctx->Input(0);
      T size = ctx->InputShape(0).num_elements();
      auto iota =
          xla::Iota(ctx->builder(),
                    xla::primitive_util::NativeToPrimitiveType<T>(), size);
      auto result = XlaScatter(iota, iota, indices,
                               /*indices_are_vectors=*/false,
                               /*indices_are_sorted=*/false,
                               /*combiner=*/{}, ctx->builder());
      OP_REQUIRES_OK(ctx, result.status());
      ctx->SetOutput(0, result.value());
    }
  }
};

REGISTER_XLA_OP(
    Name("InvertPermutation").TypeConstraint("T", {DT_INT32, DT_INT64}),
    InvertPermutationOp);

}  // namespace
}  // namespace tensorflow
