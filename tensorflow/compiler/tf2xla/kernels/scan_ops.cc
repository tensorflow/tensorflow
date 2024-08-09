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

#include <array>
#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

constexpr std::array<DataType, 6> kScanOpTypes = {
    {DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}};

enum class Reducer { kProduct, kSum, kLogSumExp };

class ScanOp : public XlaOpKernel {
 public:
  ScanOp(OpKernelConstruction* ctx, Reducer reducer)
      : XlaOpKernel(ctx), reducer_(reducer) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reverse", &reverse_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("exclusive", &exclusive_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape tensor_axis_shape = ctx->InputShape(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_axis_shape),
                errors::InvalidArgument("ScanOp: axis must be a scalar, not ",
                                        tensor_axis_shape.DebugString()));

    int64_t axis;
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

    std::vector<int64_t> window_strides(input_shape.dims(), 1);
    std::vector<int64_t> window_dims(input_shape.dims(), 1);
    window_dims[axis] = input_shape.dim_size(axis);

    std::vector<std::pair<int64_t, int64_t>> padding(input_shape.dims(),
                                                     {0, 0});
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
    switch (reducer_) {
      case Reducer::kSum:
        init = XlaHelpers::Zero(builder, dtype);
        reducer = ctx->GetOrCreateAdd(dtype);
        break;
      case Reducer::kProduct:
        init = XlaHelpers::One(builder, dtype);
        reducer = ctx->GetOrCreateMul(dtype);
        break;
      case Reducer::kLogSumExp:
        init = XlaHelpers::FloatLiteral(builder, dtype, -INFINITY);
        reducer = ctx->GetOrCreateLogAddExp(dtype);
        break;
    }
    auto output = xla::ReduceWindowWithGeneralPadding(
        XlaHelpers::ConvertElementType(ctx->Input(0), dtype), init, *reducer,
        window_dims, window_strides,
        /*base_dilations=*/{}, /*window_dilations=*/{}, padding);
    output = XlaHelpers::ConvertElementType(output, ctx->input_type(0));

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
  const Reducer reducer_;
  bool reverse_;
  bool exclusive_;
};

class CumsumOp : public ScanOp {
 public:
  explicit CumsumOp(OpKernelConstruction* ctx) : ScanOp(ctx, Reducer::kSum) {}
};
REGISTER_XLA_OP(Name("Cumsum")
                    .TypeConstraint("T", kScanOpTypes)
                    .CompileTimeConstantInput("axis"),
                CumsumOp);

class CumprodOp : public ScanOp {
 public:
  explicit CumprodOp(OpKernelConstruction* ctx)
      : ScanOp(ctx, Reducer::kProduct) {}
};
REGISTER_XLA_OP(Name("Cumprod")
                    .TypeConstraint("T", kScanOpTypes)
                    .CompileTimeConstantInput("axis"),
                CumprodOp);

class CumulativeLogsumexpOp : public ScanOp {
 public:
  explicit CumulativeLogsumexpOp(OpKernelConstruction* ctx)
      : ScanOp(ctx, Reducer::kLogSumExp) {}
};
REGISTER_XLA_OP(Name("CumulativeLogsumexp")
                    .TypeConstraint("T", kScanOpTypes)
                    .CompileTimeConstantInput("axis"),
                CumulativeLogsumexpOp);

}  // anonymous namespace
}  // namespace tensorflow
