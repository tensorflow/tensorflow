/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/comparison_util.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/comparators.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

class UniqueOpBase : public XlaOpKernel {
 public:
  explicit UniqueOpBase(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    DataType dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_idx", &dtype));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &idx_type_));
  }

  // Transpose a tensor by moving axis `from` into `to`.
  xla::XlaOp MoveAxis(xla::XlaOp a, int64_t from, int64_t to,
                      const xla::Shape& input_shape) {
    std::vector<int64_t> permutation;
    permutation.reserve(input_shape.rank());
    for (int64_t i = 0; i < input_shape.rank(); ++i) {
      permutation.push_back(i);
    }
    std::swap(permutation[from], permutation[to]);
    return xla::Transpose(a, permutation);
  }

  xla::XlaOp CumSumR1(XlaOpKernelContext* ctx, xla::XlaOp input, int64_t size) {
    auto init = xla::Zero(ctx->builder(), xla::S32);
    auto reducer = xla::CreateScalarAddComputation(xla::S32, ctx->builder());

    return xla::ReduceWindowWithGeneralPadding(
        input, init, reducer, {size}, {1},
        /*base_dilations=*/{}, /*window_dilations=*/{}, {{size - 1, 0}});
  }

  // RollingSelectR1 takes two arrays: `data` and `mask`. It scans this two
  // arrays in parallel and accumulates outputs into `accum`.
  //
  // For each position i, accum[i] = data[i]
  // if mask[i] = 1 or accum[i - 1] if mask[i] = 0.
  //
  // Requires mask[0] = 1, meaning that accum[i - 1] will never be accessed.
  //
  // This is implemented as an hlo while loop.
  xla::XlaOp RollingSelectR1(XlaOpKernelContext* ctx, xla::XlaOp data,
                             xla::XlaOp mask, int64_t size) {
    xla::XlaComputation cond, body;
    const xla::Shape r1_shape = xla::ShapeUtil::MakeShape(xla::S32, {size});
    const xla::Shape counter_shape = xla::ShapeUtil::MakeScalarShape(xla::S32);
    const xla::Shape& single_element_shape = counter_shape;

    auto loop_shape = xla::ShapeUtil::MakeTupleShape(
        {counter_shape, r1_shape, r1_shape, r1_shape});
    {
      std::unique_ptr<xla::XlaBuilder> builder =
          ctx->builder()->CreateSubBuilder("loop_cond");
      auto param = xla::Parameter(builder.get(), 0, loop_shape, "param");
      auto counter = xla::GetTupleElement(param, 0);
      auto limit = xla::ConstantR0<int32_t>(builder.get(), size);
      xla::Lt(counter, limit);

      cond = builder->Build().value();
    }

    {
      std::unique_ptr<xla::XlaBuilder> builder =
          ctx->builder()->CreateSubBuilder("loop_body");
      auto param = xla::Parameter(builder.get(), 0, loop_shape, "param");
      auto counter = xla::GetTupleElement(param, 0);

      auto data_stack = xla::GetTupleElement(param, 1);
      auto data = xla::DynamicSlice(data_stack, {counter}, {1});
      data = xla::Reshape(single_element_shape, data);

      auto mask_stack = xla::GetTupleElement(param, 2);
      auto mask = xla::DynamicSlice(mask_stack, {counter}, {1});
      mask = xla::Reshape(single_element_shape, mask);

      auto counter_minus = counter - xla::One(builder.get(), xla::S32);
      // If counter = 0, then counter_minus = 0.
      auto zero = xla::Zero(builder.get(), xla::S32);
      counter_minus = xla::Select(xla::Eq(counter, zero), zero, counter_minus);

      auto accum_stack = xla::GetTupleElement(param, 3);
      auto accum_minus = xla::DynamicSlice(accum_stack, {counter_minus}, {1});
      accum_minus = xla::Reshape(single_element_shape, accum_minus);

      auto accum = xla::Select(xla::ConvertElementType(mask, xla::PRED), data,
                               accum_minus);
      accum_stack = xla::DynamicUpdateSlice(
          accum_stack, xla::Reshape(accum, {1}), {counter});
      counter = counter + xla::One(builder.get(), xla::S32);

      xla::Tuple(builder.get(), {counter, data_stack, mask_stack, accum_stack});
      body = builder->Build().value();
    }

    auto zero = xla::Zero(ctx->builder(), xla::S32);
    auto zero_broadcast = xla::Broadcast(zero, {size});
    auto init = xla::Tuple(ctx->builder(), {zero, data, mask, zero_broadcast});
    return xla::GetTupleElement(xla::While(cond, body, init), 3);
  }

  void CompileWithAxis(XlaOpKernelContext* ctx, int64_t axis) {
    xla::XlaOp input = ctx->Input(0);
    absl::StatusOr<xla::Shape> input_shape_or = ctx->builder()->GetShape(input);
    OP_REQUIRES_OK(ctx, input_shape_or.status());
    auto input_shape = input_shape_or.value();
    axis = axis < 0 ? axis + input_shape.rank() : axis;
    OP_REQUIRES(ctx, 0 <= axis && axis < input_shape.rank(),
                errors::InvalidArgument("axis has to be between [0, ",
                                        input_shape.rank(), ")"));
    auto aux = MoveAxis(input, axis, 0, input_shape);
    auto aux_shape = ctx->builder()->GetShape(aux).value();
    int64_t leading_size = aux_shape.dimensions(0);
    int64_t product = 1;
    for (int64_t i = 1; i < aux_shape.rank(); ++i) {
      product *= aux_shape.dimensions(i);
    }
    aux = xla::Reshape(aux, {leading_size, product});
    if (leading_size == 0) {
      auto result_data = xla::Reshape(aux, aux_shape.dimensions());
      result_data = MoveAxis(result_data, 0, axis, aux_shape);
      ctx->SetOutput(0, result_data);
      ctx->SetOutput(1, xla::Iota(ctx->builder(), xla::S32, leading_size));
      return;
    }
    std::vector<xla::XlaOp> sort_keys;
    sort_keys.reserve(product + 1);
    std::vector<xla::PrimitiveType> sort_types;
    sort_types.reserve(product + 1);
    for (int64_t i = 0; i < product; ++i) {
      xla::XlaOp slice = xla::SliceInDim(aux, i, i + 1, 1, 1);
      sort_keys.push_back(xla::Reshape(slice, {leading_size}));
      sort_types.push_back(input_shape.element_type());
    }
    auto iota = xla::Iota(ctx->builder(), xla::S32, leading_size);
    sort_keys.push_back(iota);
    sort_types.push_back(xla::S32);

    std::vector<std::optional<xla::XlaOp (*)(xla::XlaOp, xla::XlaOp,
                                             absl::Span<const int64_t>)>>
        generators(sort_types.size(), xla::LtTotalOrder);
    auto lt_chain = xla::CreateScalarComparisonComputation(
        "UniqueV2Lt", sort_types, generators, ctx->builder());

    auto sorted = xla::Sort(sort_keys, lt_chain, 0, /*is_stable=*/true);
    // Last element is permutation.
    xla::XlaOp perm;
    if (sort_keys.size() == 1) {
      perm = sorted;
    } else {
      perm = xla::GetTupleElement(sorted, sort_keys.size() - 1);
    }

    // Use gather to rearrange minor dimension.
    xla::GatherDimensionNumbers gather_dim_numbers;
    gather_dim_numbers.add_offset_dims(1);
    // The dimension to rewrite is the index dim.
    gather_dim_numbers.add_start_index_map(0);
    gather_dim_numbers.set_index_vector_dim(1);
    gather_dim_numbers.add_collapsed_slice_dims(0);
    auto permuted = xla::Gather(aux, perm, gather_dim_numbers, {1, product});
    // Tail is everything except for first element.
    auto tail = xla::SliceInDim(permuted, 1, leading_size, 1, 0);
    // Init is everything except for last element.
    auto init = xla::SliceInDim(permuted, 0, leading_size - 1, 1, 0);
    auto ne = xla::Compare(tail, init, xla::ComparisonDirection::kNe);
    auto reduce =
        xla::Reduce(ne, xla::ConstantR0(ctx->builder(), false),
                    CreateScalarOrComputation(xla::PRED, ctx->builder()), {1});
    auto mask = xla::ConvertElementType(reduce, xla::S32);
    mask = xla::PadInDim(mask, xla::One(ctx->builder(), xla::S32), 0, 1, 0);
    auto iperm = RollingSelectR1(ctx, perm, mask, leading_size);

    auto sort_by_iperm =
        xla::Sort({iperm, mask, perm},
                  xla::CreateScalarLtComputation({xla::S32, xla::S32, xla::S32},
                                                 ctx->builder()),
                  0,
                  /*is_stable=*/true);
    mask = xla::GetTupleElement(sort_by_iperm, 1);
    // perm_sort is used later to revert the indices back to input order.
    auto perm_sort = xla::GetTupleElement(sort_by_iperm, 2);

    auto dynamic_size = xla::ReduceAll(
        mask, xla::Zero(ctx->builder(), xla::S32),
        xla::CreateScalarAddComputation(xla::S32, ctx->builder()));
    auto mask_sort = xla::Sort(
        {mask, perm_sort},
        xla::CreateScalarGtComputation({xla::S32, xla::S32}, ctx->builder()), 0,
        /*is_stable=*/true);
    auto mask_permute = xla::GetTupleElement(mask_sort, 1);
    permuted = xla::Gather(aux, mask_permute, gather_dim_numbers, {1, product});
    auto result_data = xla::Reshape(permuted, aux_shape.dimensions());
    result_data = MoveAxis(result_data, 0, axis, aux_shape);
    result_data = xla::SetDimensionSize(result_data, dynamic_size, axis);
    ctx->SetOutput(0, result_data);
    auto imask = CumSumR1(ctx, mask, leading_size);
    imask = xla::Sub(imask, xla::One(ctx->builder(), xla::S32), {});
    auto idx = xla::GetTupleElement(
        xla::Sort({perm_sort, imask},
                  xla::CreateScalarLtComputation({xla::S32, xla::S32},
                                                 ctx->builder())),
        1);
    idx = xla::ConvertElementType(idx, idx_type_);
    ctx->SetOutput(1, idx);
  }

 private:
  xla::PrimitiveType idx_type_;
};

class UniqueOp : public UniqueOpBase {
 public:
  explicit UniqueOp(OpKernelConstruction* ctx) : UniqueOpBase(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    CompileWithAxis(ctx, /*axis=*/0);
  }
};

REGISTER_XLA_OP(Name("Unique"), UniqueOp);

class UniqueV2Op : public UniqueOpBase {
 public:
  explicit UniqueV2Op(OpKernelConstruction* ctx) : UniqueOpBase(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<int64_t> axises;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &axises));
    OP_REQUIRES(
        ctx, axises.size() <= 1,
        xla::InvalidArgument("Only single axis unique op is supported"));
    int64_t axis;
    if (axises.empty()) {
      axis = 0;
    } else {
      axis = axises.front();
    }
    CompileWithAxis(ctx, /*axis=*/axis);
  }
};

REGISTER_XLA_OP(Name("UniqueV2").CompileTimeConstantInput("axis"), UniqueV2Op);

}  // namespace
}  // namespace tensorflow
