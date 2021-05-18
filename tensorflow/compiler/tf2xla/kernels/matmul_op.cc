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

// XLA-specific MatMul Op.

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

constexpr std::array<DataType, 10> kMatmulTypes = {
    {DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
     DT_INT32, DT_INT64, DT_INT16, DT_INT8}};

class MatMulOp : public XlaOpKernel {
 public:
  explicit MatMulOp(OpKernelConstruction* ctx, bool is_sparse = false)
      : XlaOpKernel(ctx), is_sparse_(is_sparse) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    if (is_sparse) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("Ta", &a_type_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("Tb", &b_type_));
      // SparseMatMul is actually dense matmul with a hint that one or
      // both of the inputs may contain a lot of zeroes. On CPU these
      // inputs are dynamically converted to sparse representation
      // before multiplication. For now in XLA we ignore the hints
      // and always do dense multiplication.
      bool dummy_is_sparse;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("a_is_sparse", &dummy_is_sparse));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("b_is_sparse", &dummy_is_sparse));
    }
  }

  ~MatMulOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape a_shape = ctx->InputShape(0);
    const TensorShape b_shape = ctx->InputShape(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(a_shape),
        errors::InvalidArgument("In[0] is not a matrix. Instead it has shape ",
                                a_shape.DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(b_shape),
        errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                b_shape.DebugString()));
    int first_index = transpose_a_ ? 0 : 1;
    int second_index = transpose_b_ ? 1 : 0;

    OP_REQUIRES(ctx,
                a_shape.dim_size(first_index) == b_shape.dim_size(second_index),
                errors::InvalidArgument(
                    "Matrix size-incompatible: In[0]: ", a_shape.DebugString(),
                    ", In[1]: ", b_shape.DebugString()));

    xla::XlaOp a = ctx->Input(0);
    xla::XlaOp b = ctx->Input(1);
    if (is_sparse_) {
      if (a_type_ == DT_BFLOAT16) {
        a = xla::ConvertElementType(a, xla::F32);
      }
      if (b_type_ == DT_BFLOAT16) {
        b = xla::ConvertElementType(b, xla::F32);
      }
    }
    ctx->SetOutput(0, xla::BatchDot(a, transpose_a_, b, transpose_b_));
  }

 private:
  bool is_sparse_;
  bool transpose_a_;
  bool transpose_b_;
  DataType a_type_;
  DataType b_type_;
};

REGISTER_XLA_OP(Name("MatMul").TypeConstraint("T", kMatmulTypes), MatMulOp);

class SparseMatMulOp : public MatMulOp {
 public:
  explicit SparseMatMulOp(OpKernelConstruction* ctx) : MatMulOp(ctx, true) {}

  ~SparseMatMulOp() override = default;
};

REGISTER_XLA_OP(Name("SparseMatMul"), SparseMatMulOp);

}  // namespace
}  // namespace tensorflow
