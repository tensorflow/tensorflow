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

// XLA-specific reshape Op.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class ReshapeOp : public XlaOpKernel {
 public:
  explicit ReshapeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape input_shape = ctx->InputShape(0);
    auto input_xla_shape = ctx->InputXlaShape(0);
    const TensorShape sizes_shape = ctx->InputShape(1);
    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(sizes_shape),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes_shape.DebugString()));
    const int64_t num_dims = sizes_shape.num_elements();

    std::vector<int64_t> shape_input;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntVector(
                       1, &shape_input, xla::ValueInferenceMode::kUpperBound));
    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one if there
    // is one.
    TensorShape shape;
    int64_t product = 1;
    int unknown_index = -1;
    bool shape_has_zero_dim = false;
    for (int d = 0; d < num_dims; ++d) {
      const int64_t size = shape_input[d];
      if (size == -1) {
        OP_REQUIRES(
            ctx, unknown_index == -1,
            errors::InvalidArgument("only one input size may be -1, not both ",
                                    unknown_index, " and ", d));
        unknown_index = d;
        shape.AddDim(1);
      } else if (size == 0) {
        // We don't include zero-sized dimension in product, so that we can
        // still calculate number of elements for non-zero-sized dimensions and
        // therefore infer their shapes.
        shape.AddDim(size);
        shape_has_zero_dim = true;
      } else {
        OP_REQUIRES(ctx, size >= 0,
                    errors::InvalidArgument(
                        "size ", d, " must be non-negative, not ", size));
        shape.AddDim(size);
        product *= size;
      }
    }
    auto input = ctx->Input(0);
    if (unknown_index != -1) {
      int64_t input_num_elements = 1;
      bool input_has_zero_dim = false;
      for (int dim = 0; dim < input_shape.dims(); dim++) {
        // For zero dimension, we don't count it into `input_num_elements`
        // unless `sizes` has no zero dimension, so we are still able to
        // infer shapes for other dimensions.
        if (input_shape.dim_size(dim) > 0 || !shape_has_zero_dim) {
          input_num_elements *= input_shape.dim_size(dim);
        } else {
          input_has_zero_dim = true;
        }
      }

      int64_t missing = input_num_elements / product;
      if (!input_has_zero_dim) {
        if (input_xla_shape->is_static() || input_xla_shape->rank() != 1) {
          OP_REQUIRES(
              ctx, product * missing == input_num_elements,
              errors::InvalidArgument(
                  "Input to reshape is a tensor with ", input_num_elements,
                  " values, but the requested shape requires a multiple of ",
                  product));
        } else {
          // For 1D shape, we can safely insert extra padding in the end to make
          // sure the input is multiple of the product of the known dimensions.
          // (We can probably do that for >1D shapes but that involves
          // factorizing the number of missing elements.)
          int64_t padded_input_num =
              xla::CeilOfRatio(input_num_elements, product) * product;
          missing = padded_input_num / product;
          input = xla::PadInDim(
              input, xla::Zero(ctx->builder(), input_xla_shape->element_type()),
              0, 0, padded_input_num - input_num_elements);
          input_shape.set_dim(0, padded_input_num);
        }
      }
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(ctx, shape.num_elements() == input_shape.num_elements(),
                errors::InvalidArgument("Input to reshape is a tensor with ",
                                        input_shape.num_elements(),
                                        " values, but the requested shape has ",
                                        shape.num_elements()));

    VLOG(2) << "Reshape from " << input_shape.DebugString() << " to "
            << shape.DebugString() << ", unknown_index=" << unknown_index;
    if (input_xla_shape->is_static()) {
      ctx->SetOutput(0, xla::Reshape(input, shape.dim_sizes()));
      return;
    }

    std::vector<xla::XlaOp> output_dim_sizes;
    std::vector<bool> dims_are_dynamic;
    const auto& dims = shape.dims();
    dims_are_dynamic.reserve(dims);
    for (int64_t i = 0; i < dims; ++i) {
      output_dim_sizes.push_back(
          xla::Reshape(xla::Slice(ctx->Input(1), {i}, {i + 1}, {1}), {}));
    }
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPredVector(1, &dims_are_dynamic));
    if (unknown_index == -1) {
      // No unknown index.
      ctx->SetOutput(
          0, xla::DynamicReshape(input, output_dim_sizes, shape.dim_sizes(),
                                 dims_are_dynamic));
      return;
    }
    auto common_factors =
        xla::CommonFactors(input_shape.dim_sizes(), shape.dim_sizes());

    // Find common_factors that the input belongs to.
    for (int64_t i = 0; i < common_factors.size() - 1; ++i) {
      auto start = common_factors[i];
      auto end = common_factors[i + 1];
      bool input_is_dynamic = false;
      // product of all input dims in this group. E.g., in
      // reshape(Tensor([2, 3, 3]), [3, -1, 3]) product of the group
      // containing -1 will be 6.
      xla::XlaOp product = xla::One(ctx->builder(), xla::S32);
      for (int64_t dim = start.first; dim < end.first; ++dim) {
        if (input_xla_shape->is_dynamic_dimension(dim)) {
          input_is_dynamic = true;
        }
        product = xla::Mul(product, xla::GetDimensionSize(input, dim));
      }
      bool unknown_dim_in_group = false;
      // The real size for the -1 dimension in a reshape. E.g., in
      // reshape(Tensor([2, 3, 3]), [3, -1, 3]) this will be 2.
      xla::XlaOp unknown_dim_size = product;
      for (int64_t dim = start.second; dim < end.second; ++dim) {
        if (dim == unknown_index) {
          unknown_dim_in_group = true;
        } else {
          unknown_dim_size = xla::Div(unknown_dim_size, output_dim_sizes[dim]);
        }
      }

      if (unknown_dim_in_group) {
        // If input dim is dynamic, output dim at the -1 position must be
        // dynamic. Similarly, if input dim is static, output dim has to be
        // static at the -1 dimension.
        dims_are_dynamic[unknown_index] = input_is_dynamic;
        output_dim_sizes[unknown_index] = unknown_dim_size;

        ctx->SetOutput(
            0, xla::DynamicReshape(input, output_dim_sizes, shape.dim_sizes(),
                                   dims_are_dynamic));
        VLOG(2) << "Reshape from " << ctx->InputXlaShape(0)->ToString()
                << " to " << xla::VectorString(shape.dim_sizes())
                << ", dynamic_dims=" << xla::VectorString(dims_are_dynamic);
        return;
      }
    }
  }
};

REGISTER_XLA_OP(Name("Reshape").CompileTimeConstantInput("shape"), ReshapeOp);

}  // namespace
}  // namespace tensorflow
