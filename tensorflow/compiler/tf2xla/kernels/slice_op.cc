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
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
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

    const int input_dims = input_shape.dims();
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsVector(begin_tensor_shape) &&
            TensorShapeUtils::IsVector(size_tensor_shape) &&
            begin_tensor_shape.num_elements() == input_dims &&
            size_tensor_shape.num_elements() == input_dims,
        errors::InvalidArgument(
            "Expected begin and size arguments to be 1-D tensors of size ",
            input_dims, ", but got shapes ", begin_tensor_shape.DebugString(),
            " and ", size_tensor_shape.DebugString(), " instead."));

    std::vector<int64_t> begin;
    std::vector<int64_t> size;
    const bool all_begins_are_constant =
        ctx->ConstantInputAsIntVector(1, &begin).ok();
    const bool all_sizes_are_constant =
        ctx->ConstantInputAsIntVector(2, &size).ok();
    if (all_begins_are_constant && all_sizes_are_constant) {
      std::vector<int64_t> wrapped_size(size.size());
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
        int64_t b = begin[i];
        int64_t s = wrapped_size[i];
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

      std::vector<int64_t> limits;
      limits.reserve(begin.size());
      for (int i = 0; i < begin.size(); ++i) {
        limits.push_back(begin[i] + wrapped_size[i]);
      }
      std::vector<int64_t> strides(begin.size(), 1);
      auto slice = xla::Slice(ctx->Input(0), begin, limits, strides);
      // Check for slice on dynamic dimensions.
      std::vector<bool> size_is_dynamic;
      OP_REQUIRES_OK(
          ctx, ctx->ResolveInputDynamismIntoPredVector(2, &size_is_dynamic));

      for (int64_t i = 0; i < size.size(); ++i) {
        if (size_is_dynamic[i]) {
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
      // When a size is -1, we take rest of the dimension according to
      // https://www.tensorflow.org/api_docs/python/tf/slice.
      // This essentially makes size as dynamic.
      bool constant_size_is_minus_one = false;
      // `begin` or `size` is not a compile-time constant.
      if (all_sizes_are_constant) {
        for (int i = 0; i < input_dims; ++i) {
          if (size[i] < 0) {
            OP_REQUIRES(ctx, size[i] == -1,
                        errors::InvalidArgument(
                            "Negative size of slice operator can only be -1"));
            constant_size_is_minus_one = true;
          }

          OP_REQUIRES(ctx, size[i] <= input_shape.dim_size(i),
                      errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                              input_shape.dim_size(i),
                                              "], but ", "got ", size[i]));
        }
      }

      absl::InlinedVector<xla::XlaOp, 4> begin_indices;
      begin_indices.reserve(input_dims);
      xla::XlaOp begin = ctx->Input("begin");
      for (int i = 0; i < input_dims; i++) {
        begin_indices.push_back(
            xla::Reshape(xla::Slice(begin, {i}, {i + 1}, {1}), {}));
      }
      if (all_sizes_are_constant && !constant_size_is_minus_one) {
        xla::XlaOp input = ctx->Input(0);
        ctx->SetOutput(0, xla::DynamicSlice(input, begin_indices, size));
      } else {
        // Size is not constant, use input size as upperbound and then set
        // dimension size on it.

        // First pad input with input size to avoid OOB -- dynamic slice with
        // OOB slice produces undesired results.
        xla::PaddingConfig padding_config;
        xla::XlaOp input = ctx->Input(0);
        for (int64_t i = 0; i < input_dims; ++i) {
          auto* dims = padding_config.add_dimensions();
          dims->set_edge_padding_low(0);
          dims->set_edge_padding_high(input_shape.dim_size(i));
          dims->set_interior_padding(0);
          input = xla::RemoveDynamicDimension(input, i);
        }
        auto padded_input =
            xla::Pad(input, xla::Zero(ctx->builder(), ctx->input_xla_type(0)),
                     padding_config);
        // Slice full size out of the input starting from the offsets.
        auto sliced = xla::DynamicSlice(padded_input, begin_indices,
                                        input_shape.dim_sizes());
        for (int i = 0; i < input_dims; i++) {
          xla::XlaOp dynamic_size =
              xla::Reshape(xla::Slice(ctx->Input(2), {i}, {i + 1}, {1}), {});
          if (constant_size_is_minus_one && size[i] == -1) {
            // size = input_.dim_size(i) - begin[i]
            dynamic_size = xla::ConstantR0<int32>(ctx->builder(),
                                                  input_shape.dim_size(i)) -
                           begin_indices[i];
          }
          auto constant_size = ctx->value_inference().AnalyzeConstant(
              dynamic_size, xla::ValueInferenceMode::kValue);
          OP_REQUIRES_OK(ctx, constant_size.status());
          if (constant_size->AllValid()) {
            // Slice size on this dimension is constant. This branch is
            // triggered when some dimensions's slice sizes are constant while
            // some are dynamic.
            sliced = xla::SliceInDim(
                sliced, 0, constant_size->Get<int32>({}).value(), 1, i);
          } else {
            // We gave a generous bound (same as input) to the output, try reset
            // the bound if a tighter one can be found.
            auto status = xla::SetDimensionSizeWithRebound(
                &ctx->value_inference(), sliced, dynamic_size, i);
            OP_REQUIRES_OK(ctx, status.status());
            sliced = status.ValueOrDie();
          }
        }
        ctx->SetOutput(0, sliced);
      }
    }
  }
};

REGISTER_XLA_OP(Name("Slice")
                    .CompileTimeConstantInput("begin")
                    .CompileTimeConstantInput("size"),
                SliceOp);

}  // namespace
}  // namespace tensorflow
