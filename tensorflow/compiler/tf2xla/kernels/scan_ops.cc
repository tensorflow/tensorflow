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

#include <vector>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// TODO(phawkins): implement double-sized windowed reductions in XLA and remove
// the type constraint.
constexpr std::array<DataType, 3> kScanOpTypes = {
    {DT_HALF, DT_BFLOAT16, DT_FLOAT}};

class ScanOp : public XlaOpKernel {
 public:
  ScanOp(OpKernelConstruction* ctx, bool sum) : XlaOpKernel(ctx), sum_(sum) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reverse", &reverse_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("exclusive", &exclusive_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape tensor_axis_shape = ctx->InputShape(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_axis_shape),
                errors::InvalidArgument("ScanOp: axis must be a scalar, not ",
                                        tensor_axis_shape.DebugString()));

    int64 axis;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &axis));
    if (axis < 0) {
      axis += input_shape.dims();
    }
    OP_REQUIRES(
        ctx, FastBoundsCheck(axis, input_shape.dims()),
        errors::InvalidArgument("ScanOp: Expected scan axis in the range [",
                                -input_shape.dims(), ", ", input_shape.dims(),
                                "), but got ", axis));

    DataType dtype = XlaHelpers::SumAccumulationType(ctx->input_type(0));

    if (input_shape.num_elements() == 0) {
      // Exit early if there is nothing to compute.
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }

    xla::XlaBuilder* builder = ctx->builder();

    std::vector<int64> window_strides(input_shape.dims(), 1);
    std::vector<int64> window_dims(input_shape.dims(), 1);
    window_dims[axis] = input_shape.dim_size(axis);

    std::vector<std::pair<int64, int64>> padding(input_shape.dims(), {0, 0});
    padding[axis].first = input_shape.dim_size(axis) - 1;
    // In exclusive mode, add an extra padding element so there is a complete
    // window of padding before the data starts.
    if (exclusive_) {
      ++padding[axis].first;
    }
    if (reverse_) {
      std::swap(padding[axis].first, padding[axis].second);
    }

    xla::XlaOp init;
    const xla::XlaComputation* reducer;
    if (sum_) {
      init = XlaHelpers::Zero(builder, dtype);
      reducer = ctx->GetOrCreateAdd(dtype);
    } else {
      init = XlaHelpers::One(builder, dtype);
      reducer = ctx->GetOrCreateMul(dtype);
    }
    auto output = xla::ReduceWindowWithGeneralPadding(
        XlaHelpers::ConvertElementType(builder, ctx->Input(0), dtype), init,
        *reducer, window_dims, window_strides, padding);
    output =
        XlaHelpers::ConvertElementType(builder, output, ctx->input_type(0));

    // In exclusive mode, we have computed an extra element containing the sum
    // of all the input elements. Slice off this extra "last" element.
    if (exclusive_) {
      if (reverse_) {
        output =
            xla::SliceInDim(output, 1, input_shape.dim_size(axis) + 1, 1, axis);

      } else {
        output =
            xla::SliceInDim(output, 0, input_shape.dim_size(axis), 1, axis);
      }
    }
    ctx->SetOutput(0, output);
  }

 private:
  const bool sum_;  // True=cumulative sum. False=cumulative product.
  bool reverse_;
  bool exclusive_;
};

class CumsumOp : public ScanOp {
 public:
  explicit CumsumOp(OpKernelConstruction* ctx) : ScanOp(ctx, /*sum=*/true) {}
};
REGISTER_XLA_OP(Name("Cumsum")
                    .TypeConstraint("T", kScanOpTypes)
                    .CompileTimeConstInput("axis"),
                CumsumOp);

class CumprodOp : public ScanOp {
 public:
  explicit CumprodOp(OpKernelConstruction* ctx) : ScanOp(ctx, /*sum=*/false) {}
};
REGISTER_XLA_OP(Name("Cumprod")
                    .TypeConstraint("T", kScanOpTypes)
                    .CompileTimeConstInput("axis"),
                CumprodOp);

}  // anonymous namespace
}  // namespace tensorflow
