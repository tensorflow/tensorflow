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

// XLA-specific Slice Op.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {
namespace {

class SliceOp : public XlaOpKernel {
 public:
  explicit SliceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    bool is_identity = true;
    std::vector<int64> begin;
    std::vector<int64> size;
    SharedValidation(ctx, &is_identity, &begin, &size);
    if (!ctx->status().ok()) return;

    if (is_identity) {
      VLOG(1) << "Slice identity";
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }

    // slice will be an empty handle if the output has no elements.
    CHECK_EQ(begin.size(), size.size());
    std::vector<int64> limits;
    for (int i = 0; i < begin.size(); ++i) {
      limits.push_back(begin[i] + size[i]);
    }
    ctx->SetOutput(0, ctx->builder()->Slice(ctx->Input(0), begin, limits));
  }

 private:
  void SharedValidation(XlaOpKernelContext* ctx, bool* is_identity,
                        std::vector<int64>* begin, std::vector<int64>* size);
};

void SliceOp::SharedValidation(XlaOpKernelContext* ctx, bool* is_identity,
                               std::vector<int64>* begin,
                               std::vector<int64>* size) {
  const TensorShape input_shape = ctx->InputShape(0);
  const TensorShape begin_tensor_shape = ctx->InputShape(1);
  const TensorShape size_tensor_shape = ctx->InputShape(2);

  OP_REQUIRES(
      ctx,
      IsLegacyVector(begin_tensor_shape) && IsLegacyVector(size_tensor_shape) &&
          begin_tensor_shape.num_elements() == input_shape.dims() &&
          size_tensor_shape.num_elements() == input_shape.dims(),
      errors::InvalidArgument(
          "Expected begin and size arguments to be 1-D tensors of size ",
          input_shape.dims(), ", but got shapes ",
          begin_tensor_shape.DebugString(), " and ",
          size_tensor_shape.DebugString(), " instead."));

  const int input_dims = input_shape.dims();

  OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, begin));
  OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(2, size));
  for (int i = 0; i < input_dims; ++i) {
    if ((*size)[i] == -1) {
      // A size[i] of -1 means "all elements from begin[i] to dim_size(i)".
      (*size)[i] = input_shape.dim_size(i) - (*begin)[i];
    }
  }

  *is_identity = true;
  for (int i = 0; i < input_dims; ++i) {
    int64 b = (*begin)[i];
    int64 s = (*size)[i];
    if (input_shape.dim_size(i) == 0) {
      OP_REQUIRES(ctx, b == 0 && s == 0,
                  errors::InvalidArgument(
                      "Expected begin[", i, "] == 0 (got ", b, ") and size[", i,
                      "] == 0 ", "(got ", s, ") when ", "input_shape.dim_size(",
                      i, ") == 0"));
    } else {
      OP_REQUIRES(
          ctx, 0 <= b && b <= input_shape.dim_size(i),
          errors::InvalidArgument("Expected begin[", i, "] in [0, ",
                                  input_shape.dim_size(i), "], but got ", b));
      OP_REQUIRES(ctx, 0 <= s && b + s <= input_shape.dim_size(i),
                  errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                          input_shape.dim_size(i) - b,
                                          "], but ", "got ", s));
    }
    const bool take_all = (b == 0) && (s == input_shape.dim_size(i));
    (*is_identity) &= take_all;
  }
}

REGISTER_XLA_OP("Slice", SliceOp);

}  // namespace
}  // namespace tensorflow
