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

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {
namespace {

class SliceOp : public XlaOpKernel {
 public:
  explicit SliceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape begin_tensor_shape = ctx->InputShape(1);
    const TensorShape size_tensor_shape = ctx->InputShape(2);

    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsVector(begin_tensor_shape) &&
            TensorShapeUtils::IsVector(size_tensor_shape) &&
            begin_tensor_shape.num_elements() == input_shape.dims() &&
            size_tensor_shape.num_elements() == input_shape.dims(),
        errors::InvalidArgument(
            "Expected begin and size arguments to be 1-D tensors of size ",
            input_shape.dims(), ", but got shapes ",
            begin_tensor_shape.DebugString(), " and ",
            size_tensor_shape.DebugString(), " instead."));

    const int input_dims = input_shape.dims();

    std::vector<int64> begin;
    std::vector<int64> size;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(2, &size));
    std::vector<int64> wrapped_size(size.size());
    if (ctx->ConstantInputAsIntVector(1, &begin).ok()) {
      // `begin` is a compile-time constant.
      for (int i = 0; i < input_dims; ++i) {
        if (size[i] == -1) {
          // A size[i] of -1 means "all elements from begin[i] to dim_size(i)".
          wrapped_size[i] = input_shape.dim_size(i) - begin[i];
        } else {
          wrapped_size[i] = size[i];
        }
      }

      for (int i = 0; i < input_dims; ++i) {
        int64 b = begin[i];
        int64 s = wrapped_size[i];
        if (input_shape.dim_size(i) == 0) {
          OP_REQUIRES(ctx, b == 0 && s == 0,
                      errors::InvalidArgument(
                          "Expected begin[", i, "] == 0 (got ", b,
                          ") and size[", i, "] == 0 ", "(got ", s, ") when ",
                          "input_shape.dim_size(", i, ") == 0"));
        } else {
          OP_REQUIRES(ctx, 0 <= b && b <= input_shape.dim_size(i),
                      errors::InvalidArgument("Expected begin[", i, "] in [0, ",
                                              input_shape.dim_size(i),
                                              "], but got ", b));
          OP_REQUIRES(ctx, 0 <= s && b + s <= input_shape.dim_size(i),
                      errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                              input_shape.dim_size(i) - b,
                                              "], but ", "got ", s));
        }
      }

      std::vector<int64> limits;
      limits.reserve(begin.size());
      for (int i = 0; i < begin.size(); ++i) {
        limits.push_back(begin[i] + wrapped_size[i]);
      }
      std::vector<int64> strides(begin.size(), 1);
      auto slice = xla::Slice(ctx->Input(0), begin, limits, strides);
      // Check for slice on dynamic dimensions.
      ctx->set_dynamic_dimension_is_minus_one(true);
      std::vector<int64> dynamic_size;
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(2, &dynamic_size));

      for (int64 i = 0; i < size.size(); ++i) {
        if (dynamic_size[i] == -1) {
          if (size[i] != -1) {
            // If there is a dynamic dimension, properly set dimension size of
            // the slice.
            auto dynamic_size =
                xla::Reshape(xla::Slice(ctx->Input(2), {i}, {i + 1}, {1}), {});

            slice = xla::SetDimensionSize(slice, dynamic_size, i);
          }
        }
      }
      ctx->SetOutput(0, slice);
    } else {
      // `begin` is not a compile-time constant.
      for (int i = 0; i < input_dims; ++i) {
        OP_REQUIRES(ctx, 0 <= size[i],
                    errors::InvalidArgument(
                        "XLA compilation of Slice operator with negative sizes "
                        "requires that 'begin' is a compile-time constant."));
        OP_REQUIRES(ctx, size[i] <= input_shape.dim_size(i),
                    errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                            input_shape.dim_size(i), "], but ",
                                            "got ", size[i]));
      }
      ctx->SetOutput(0, xla::DynamicSlice(ctx->Input(0), ctx->Input(1), size));
    }
  }
};

REGISTER_XLA_OP(Name("Slice")
                    .CompileTimeConstantInput("begin")
                    .CompileTimeConstantInput("size"),
                SliceOp);

}  // namespace
}  // namespace tensorflow
