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

// XLA-specific Ops for split.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class SplitOp : public XlaOpKernel {
 public:
  explicit SplitOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const int32 num_split = num_outputs();
    const TensorShape split_dim_shape = ctx->InputShape("split_dim");
    const TensorShape input_shape = ctx->InputShape(1);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(split_dim_shape),
        errors::InvalidArgument("split_dim must be a scalar but has rank ",
                                split_dim_shape.dims()));
    int64 split_dim_orig;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(0, &split_dim_orig));

    int32 split_dim = split_dim_orig < 0 ? split_dim_orig + input_shape.dims()
                                         : split_dim_orig;
    OP_REQUIRES(ctx, 0 <= split_dim && split_dim < input_shape.dims(),
                errors::InvalidArgument("-input rank(-", input_shape.dims(),
                                        ") <= split_dim < input rank (",
                                        input_shape.dims(), "), but got ",
                                        split_dim_orig));

    OP_REQUIRES(
        ctx, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    OP_REQUIRES(
        ctx, input_shape.dim_size(split_dim) % num_split == 0,
        errors::InvalidArgument(
            "Number of ways to split should evenly divide the split "
            "dimension, but got split_dim ",
            split_dim_orig, " (size = ", input_shape.dim_size(split_dim), ") ",
            "and num_split ", num_split));

    // All the slices are the same size: this is the size along the
    // split dimension.
    const int32 slice_size = input_shape.dim_size(split_dim) / num_split;

    // The vectors we will use to define the slice. The entry for the
    // split dimensions varies for each output.
    std::vector<int64> begin(input_shape.dims(), 0);
    std::vector<int64> limits(input_shape.dims());
    std::vector<int64> strides(input_shape.dims(), 1);
    for (int i = 0; i < input_shape.dims(); ++i) {
      // Initially set up the limits to be the full size of the input:
      // the split dimension is filled in below.
      int64 dim = input_shape.dim_size(i);
      limits[i] = dim;
    }

    auto input = ctx->Input(1);

    // Create each of the outputs.
    for (int i = 0; i < num_split; ++i) {
      // Slice out the ith split from the split dimension.
      begin[split_dim] = i * slice_size;
      limits[split_dim] = (i + 1) * slice_size;
      ctx->SetOutput(i, xla::Slice(input, begin, limits, strides));
    }
  }
};

REGISTER_XLA_OP(Name("Split").CompileTimeConstantInput("split_dim"), SplitOp);

class SplitVOp : public XlaOpKernel {
 public:
  explicit SplitVOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const int32 num_split = num_outputs();
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape index_shape = ctx->InputShape(2);

    OP_REQUIRES(ctx, index_shape.num_elements() == 1,
                errors::InvalidArgument(
                    "split_dim_tensor must have exactly one element."));

    int64 split_dim_orig;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(2, &split_dim_orig));
    int64 split_dim = split_dim_orig < 0 ? split_dim_orig + input_shape.dims()
                                         : split_dim_orig;
    OP_REQUIRES(ctx, 0 <= split_dim && split_dim < input_shape.dims(),
                errors::InvalidArgument("-input rank(-", input_shape.dims(),
                                        ") <= split_dim < input rank (",
                                        input_shape.dims(), "), but got ",
                                        split_dim_orig));

    xla::XlaOp input = ctx->Input(0);

    OP_REQUIRES(ctx, input_shape.dims() > 0,
                errors::InvalidArgument("Can't split a 0 dimensional input"));

    OP_REQUIRES(
        ctx, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    // Check that sizes are correct.
    int total_split_size = 0;
    int neg_one_dim = -1;
    const TensorShape split_size_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx,
                split_size_shape.dims() == 1 &&
                    split_size_shape.num_elements() == num_split,
                errors::InvalidArgument(
                    "shape of tensor describing "
                    " the output must have dimension 1 and the same "
                    " number of elements as the output. Got ",
                    split_size_shape.dims(), "-D and ",
                    split_size_shape.num_elements(), " elements"));
    // Get the dimension of this split.
    std::vector<int64> split_sizes;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &split_sizes));

    for (int i = 0; i < num_split; ++i) {
      int64 slice_size = split_sizes[i];
      if (slice_size == -1) {
        OP_REQUIRES(
            ctx, neg_one_dim == -1,
            errors::InvalidArgument("Only one dimensions can have a value of"
                                    "-1. Second one found at dimension ",
                                    i));
        neg_one_dim = i;
      } else {
        total_split_size += slice_size;
      }
    }

    OP_REQUIRES(
        ctx,
        (neg_one_dim == -1 &&
         total_split_size == input_shape.dim_size(split_dim)) ||
            (neg_one_dim >= 0 &&
             total_split_size <= input_shape.dim_size(split_dim)),
        errors::InvalidArgument("Determined shape must either match "
                                "input shape along split_dim exactly if "
                                "fully specified, or be less than the size of "
                                "the input along split_dim if not fully "
                                "specified.  Got: ",
                                total_split_size));

    if (neg_one_dim >= 0) {
      split_sizes[neg_one_dim] =
          input_shape.dim_size(split_dim) - total_split_size;
    }

    // The vectors we will use to define the slice. The entry for the
    // split dimensions varies for each output.
    std::vector<int64> begin(input_shape.dims(), 0);
    auto dim_sizes = input_shape.dim_sizes();
    std::vector<int64> limits(dim_sizes.begin(), dim_sizes.end());
    std::vector<int64> strides(input_shape.dims(), 1);
    for (int i = 0; i < num_split; ++i) {
      TensorShape output_shape(input_shape);
      int slice_size = split_sizes[i];
      output_shape.set_dim(split_dim, slice_size);

      // Slice out the ith split from the split dimension.
      limits[split_dim] = begin[split_dim] + slice_size;
      ctx->SetOutput(i, xla::Slice(input, begin, limits, strides));
      begin[split_dim] = limits[split_dim];
    }
  }
};

REGISTER_XLA_OP(Name("SplitV")
                    .CompileTimeConstantInput("split_dim")
                    .CompileTimeConstantInput("size_splits"),
                SplitVOp);

}  // namespace
}  // namespace tensorflow
