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

#include "tensorflow/core/util/strided_slice_op.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {
namespace {
using errors::InvalidArgument;

class StridedSliceOp : public XlaOpKernel {
 public:
  explicit StridedSliceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("begin_mask", &begin_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("end_mask", &end_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ellipsis_mask", &ellipsis_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("new_axis_mask", &new_axis_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shrink_axis_mask", &shrink_axis_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Index", &index_type_));
  }

  void EmitDynamicSlice(XlaOpKernelContext* ctx,
                        const absl::InlinedVector<int64_t, 4>& strides,
                        PartialTensorShape partial_processing_shape,
                        PartialTensorShape partial_final_shape,
                        const StridedSliceShapeSpec& shape_spec,
                        const std::vector<bool>& begins_are_dynamic,
                        const std::vector<bool>& ends_are_dynamic) {
    const TensorShape input_shape = ctx->InputShape(0);
    xla::XlaOp slice = ctx->Input(0);
    for (int64_t i = 0; i < ctx->InputShape("begin").dims(); ++i) {
      OP_REQUIRES(ctx, strides[i] == 1,
                  errors::InvalidArgument(
                      "Strides have to be one when inputs are not constant."));
    }
    // Infer static output shape, reconcile unknown dimension with input dim
    // size.
    for (int64_t i = 0; i < partial_final_shape.dims(); ++i) {
      if (partial_final_shape.dim_size(i) == -1) {
        // Use input shape to update unknown dimension of partial shape -- if a
        // dimension is unknown, we use input shape as bound.
        partial_final_shape.set_dim(
            i,
            input_shape.dim_size(shape_spec.output_to_processing_mapping[i]));
      }
    }

    TensorShape final_shape;
    OP_REQUIRES(
        ctx, partial_final_shape.AsTensorShape(&final_shape),
        InvalidArgument("XLA can't deduce compile time constant output "
                        "shape for strided slice: ",
                        partial_final_shape.DebugString(),
                        ", output shape must be a compile-time constant"));
    for (int64_t i = 0; i < partial_processing_shape.dims(); ++i) {
      if (partial_processing_shape.dim_size(i) == -1) {
        // Use input shape to update unknown dimension of partial shape -- if a
        // dimension is unknown, we use input shape as bound.
        partial_processing_shape.set_dim(i, input_shape.dim_size(i));
      }
    }
    TensorShape processing_shape;
    OP_REQUIRES(
        ctx, partial_processing_shape.AsTensorShape(&processing_shape),
        InvalidArgument("XLA can't deduce compile time constant processing "
                        "shape for strided slice: ",
                        partial_processing_shape.DebugString(),
                        ", output shape must be a compile-time constant"));
    // If there is dynamic begin/end (and if the dimension is not shrunk), we
    // need to use dynamic shape infrastructure -- we slice the output with
    // full size, then call SetDimensionSize on the output. However, if we
    // slice with the full size at a non-zero dimension we may get OOB access.
    // To avoid that, we first pad the input to 2x before calling slice.
    xla::PaddingConfig padding_config;
    bool need_padding = false;
    std::vector<bool> result_dims_are_dynamic;
    const auto& dims = input_shape.dims();
    result_dims_are_dynamic.reserve(dims);
    for (int64_t i = 0; i < dims; ++i) {
      int64_t sparse_index = shape_spec.processing_to_sparse_mapping[i];
      bool shrink_axis_set = (1 << i) & shape_spec.shrink_axis_dense_mask;
      auto* dims = padding_config.add_dimensions();
      dims->set_edge_padding_low(0);

      dims->set_interior_padding(0);
      if ((begins_are_dynamic[sparse_index] ||
           ends_are_dynamic[sparse_index]) &&
          !shrink_axis_set) {
        // Need to slice this dimension so pad first.
        dims->set_edge_padding_high(input_shape.dim_size(i));
        need_padding = true;
        result_dims_are_dynamic.push_back(true);
      } else {
        dims->set_edge_padding_high(0);
        result_dims_are_dynamic.push_back(false);
      }
    }

    if (need_padding) {
      // Pad input to 2x to avoid OOB access.
      slice = xla::Pad(slice, xla::Zero(ctx->builder(), ctx->input_xla_type(0)),
                       padding_config);
      for (int64 i = 0; i < result_dims_are_dynamic.size(); ++i) {
        if (result_dims_are_dynamic[i]) {
          slice = xla::RemoveDynamicDimension(slice, i);
        }
      }
    }
    std::vector<xla::XlaOp> start_indices;
    std::vector<xla::XlaOp> slice_sizes_dynamic;
    xla::Shape input_xla_shape = ctx->InputXlaShape(0).value();
    for (int64_t i = 0; i < input_shape.dims(); ++i) {
      bool begin_mask = (1 << i) & shape_spec.begin_dense_mask;
      bool end_mask = (1 << i) & shape_spec.end_dense_mask;
      auto zero = xla::Zero(ctx->builder(), ctx->InputXlaType("begin"));
      xla::XlaOp begin_index, end_index;
      int64_t sparse_index = shape_spec.processing_to_sparse_mapping[i];
      bool xla_input_is_dynamic = input_xla_shape.is_dynamic_dimension(i);
      xla::XlaOp dim_size;
      if (xla_input_is_dynamic) {
        dim_size = xla::GetDimensionSize(ctx->Input(0), i);
        OP_REQUIRES(ctx, ctx->InputXlaType("begin") == xla::S32,
                    errors::InvalidArgument("'begin shape has to be int32 when "
                                            "indices to slice op are dynamic"));
      } else {
        dim_size =
            xla::ConstantR0WithType(ctx->builder(), ctx->InputXlaType("begin"),
                                    input_xla_shape.dimensions(i));
      }

      auto scalar_must_be_non_negative = [ctx](xla::XlaOp value) -> bool {
        // Check if the lower-bound of a value is always >= 0
        auto lower_bound = ctx->value_inference().AnalyzeConstant(
            value, xla::ValueInferenceMode::kLowerBound);
        if (!lower_bound.ok() || !lower_bound->AllValid()) {
          // Can't infer a lower bound.
          return false;
        }
        return lower_bound->Get<int32>({}) >= 0;
      };
      if (begin_mask) {
        begin_index = zero;
      } else {
        begin_index = xla::Slice(ctx->Input("begin"), {sparse_index},
                                 {sparse_index + 1}, {1});
        begin_index = xla::Reshape(begin_index, {});
        if (!scalar_must_be_non_negative(begin_index)) {
          // begin could be negative.
          auto index_negative = xla::Lt(begin_index, zero);
          auto wrapped_index = xla::Add(dim_size, begin_index);
          // Wrap negative indices around.
          begin_index = xla::Select(index_negative, wrapped_index, begin_index);
        }
      }
      start_indices.push_back(begin_index);
      if (end_mask) {
        end_index = dim_size;
      } else {
        end_index = xla::Slice(ctx->Input("end"), {sparse_index},
                               {sparse_index + 1}, {1});
        end_index = xla::Reshape(end_index, {});
        if (!scalar_must_be_non_negative(end_index)) {
          // end could be negative.
          auto index_negative = xla::Lt(end_index, zero);
          auto wrapped_index = xla::Add(dim_size, end_index);
          end_index = xla::Select(index_negative, wrapped_index, end_index);
        }
      }
      // This is safe to downcast as set dimension size  makes sure that the dim
      // in the input doesn't exceed INT32 max.
      xla::XlaOp size = xla::Max(xla::Sub(end_index, begin_index), zero);
      slice_sizes_dynamic.push_back(xla::ConvertElementType(size, xla::S32));
    }

    slice =
        xla::DynamicSlice(slice, start_indices, processing_shape.dim_sizes());
    // new_axis_mask_, ellipsis_mask_ and shrink_axis_mask_ may add or remove
    // size 1 dims of a shape.
    slice = xla::Reshape(slice, final_shape.dim_sizes());
    for (int64_t i = 0; i < final_shape.dims(); ++i) {
      int64 processing_shape_dim = shape_spec.output_to_processing_mapping[i];
      // If processing_shape_dim is -1, it means the output dimension was newly
      // added by new_axis_mask_, which doesn't show up in input.
      if (processing_shape_dim != -1) {
        // We gave a generous bounds (same as input) to the output for both
        // static and dynamic dimensions, try reset the bound if a tighter one
        // can be found.
        auto status = xla::SetDimensionSizeWithRebound(
            &ctx->value_inference(), slice,
            slice_sizes_dynamic[processing_shape_dim], i);
        OP_REQUIRES_OK(ctx, status.status());
        slice = status.value();
      }
    }
    ctx->SetOutput(0, slice);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape begin_shape = ctx->InputShape("begin");
    OP_REQUIRES(
        ctx, begin_shape.dims() == 1,
        errors::InvalidArgument("'begin' input has to be a rank 1 vector"));

    absl::InlinedVector<int64_t, 4> begin;
    absl::InlinedVector<int64_t, 4> end;
    absl::InlinedVector<int64_t, 4> strides;

    xla::Literal begin_literal, end_literal, strides_literal;
    bool begin_is_constant = ctx->ConstantInput(1, &begin_literal).ok();
    bool end_is_constant = ctx->ConstantInput(2, &end_literal).ok();
    // Strides have to be static.
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(3, &strides_literal));

    Tensor begin_tensor, end_tensor, strides_tensor;
    if (begin_is_constant) {
      OP_REQUIRES_OK(
          ctx, LiteralToHostTensor(begin_literal, index_type_, &begin_tensor));
    }
    if (end_is_constant) {
      OP_REQUIRES_OK(
          ctx, LiteralToHostTensor(end_literal, index_type_, &end_tensor));
    }
    OP_REQUIRES_OK(ctx, LiteralToHostTensor(strides_literal, index_type_,
                                            &strides_tensor));
    PartialTensorShape partial_processing_shape, partial_final_shape;
    bool dummy = false;
    StridedSliceShapeSpec shape_spec;
    OP_REQUIRES_OK(
        ctx,
        ValidateStridedSliceOp(
            begin_is_constant ? &begin_tensor : nullptr,
            end_is_constant ? &end_tensor : nullptr, strides_tensor,
            input_shape, begin_mask_, end_mask_, ellipsis_mask_, new_axis_mask_,
            shrink_axis_mask_, &partial_processing_shape, &partial_final_shape,
            &dummy, &dummy, &dummy, &begin, &end, &strides, &shape_spec));

    xla::XlaOp slice = ctx->Input(0);
    std::vector<bool> begins_are_dynamic;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPredVector(1, &begins_are_dynamic));
    std::vector<bool> ends_are_dynamic;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPredVector(2, &ends_are_dynamic));
    if (begin_is_constant && end_is_constant) {
      TensorShape final_shape;
      OP_REQUIRES(
          ctx, partial_final_shape.AsTensorShape(&final_shape),
          InvalidArgument("XLA can't deduce compile time constant output "
                          "shape for strided slice: ",
                          partial_final_shape.DebugString(),
                          ", output shape must be a compile-time constant"));
      absl::InlinedVector<int64_t, 4> dimensions_to_reverse;
      absl::InlinedVector<int64_t, 4> slice_begin, slice_end, slice_strides;
      for (int i = 0; i < begin.size(); ++i) {
        if (strides[i] > 0) {
          slice_begin.push_back(begin[i]);
          slice_end.push_back(std::max(end[i], begin[i]));
          slice_strides.push_back(strides[i]);
        } else {
          // Negative stride: swap begin and end, add 1 because the interval
          // is semi-open, and mark the dimension to be reversed.
          slice_begin.push_back(input_shape.dim_size(i) - begin[i] - 1);
          slice_end.push_back(std::max(input_shape.dim_size(i) - end[i] - 1,
                                       input_shape.dim_size(i) - begin[i] - 1));
          slice_strides.push_back(-strides[i]);
          dimensions_to_reverse.push_back(i);
        }
      }
      if (!dimensions_to_reverse.empty()) {
        slice = xla::Rev(slice, dimensions_to_reverse);
      }
      slice = xla::Slice(slice, slice_begin, slice_end, slice_strides);
      auto operand_shape_or = ctx->builder()->GetShape(ctx->Input(0));
      OP_REQUIRES_OK(ctx, operand_shape_or.status());
      xla::Shape xla_shape = operand_shape_or.value();

      bool begins_are_static = absl::c_all_of(
          begins_are_dynamic, [](bool dynamic) { return !dynamic; });
      OP_REQUIRES(ctx, begins_are_static,
                  errors::InvalidArgument(
                      "XLA can't use dynamic begin values for slice."));
      bool ends_are_static = absl::c_all_of(
          ends_are_dynamic, [](bool dynamic) { return !dynamic; });
      // Static output shape, return a static slice.
      slice = xla::Reshape(slice, final_shape.dim_sizes());
      if (xla_shape.is_static() && ends_are_static) {
        ctx->SetOutput(0, slice);
        return;
      }

      for (int64_t i = 0; i < final_shape.dims(); ++i) {
        int64_t input_index = shape_spec.output_to_processing_mapping[i];
        if (input_index == -1) {
          continue;
        }
        bool input_is_dynamic = xla_shape.is_dynamic_dimension(input_index);

        int64_t sparse_index = shape_spec.output_to_sparse_mapping[i];
        bool end_is_dynamic =
            sparse_index == -1 ? false : ends_are_dynamic[sparse_index];
        bool backward_slice = sparse_index == -1
                                  ? false
                                  : end_literal.Get<int32>({sparse_index}) < 0;
        if (input_is_dynamic || end_is_dynamic) {
          OP_REQUIRES(
              ctx, strides[input_index] == 1,
              errors::InvalidArgument("XLA has not implemented dynamic "
                                      "sized slice with non-trival stride yet. "
                                      "Please file a bug against XLA"));
          // If there is a dynamic dimension, properly set dimension size of
          // the result.
          auto operand_size = xla::GetDimensionSize(ctx->Input(0), input_index);
          if (backward_slice) {
            // We consider slicing a dynamic tensor t with negative indices as
            // a dynamic sized slice. E.g., t[: -n], the result length is
            // shape(t) - n.
            OP_REQUIRES(ctx, !end_is_dynamic,
                        errors::InvalidArgument(
                            "XLA has not implemented dynamic "
                            "sized slice with dynamic negative index %lld. "));
            operand_size = xla::Add(
                operand_size,
                xla::ConstantR0<int32>(ctx->builder(),
                                       end_literal.Get<int32>({sparse_index})));
          } else {
            // The end of slice with dynamic slice size is the min of operand
            // shape and slice size. E.g., t[:end_size], result size is
            // min(shape(t), end_size).
            xla::XlaOp end_size;
            if (end_is_dynamic) {
              end_size = xla::Reshape(xla::Slice(ctx->Input(2), {sparse_index},
                                                 {sparse_index + 1}, {1}),
                                      {});
            } else {
              end_size =
                  xla::ConstantR0<int32>(ctx->builder(), end[input_index]);
            }
            operand_size = xla::Min(operand_size, end_size);
          }
          slice = xla::SetDimensionSize(
              slice,
              xla::Sub(operand_size, xla::ConstantR0<int32>(
                                         ctx->builder(), begin[input_index])),
              i);
        }
      }
      ctx->SetOutput(0, slice);
      return;
    } else {
      EmitDynamicSlice(ctx, strides, partial_processing_shape,
                       partial_final_shape, shape_spec, begins_are_dynamic,
                       ends_are_dynamic);
    }
  }

 private:
  int32 begin_mask_, end_mask_;
  int32 ellipsis_mask_, new_axis_mask_, shrink_axis_mask_;
  DataType index_type_;
};

REGISTER_XLA_OP(Name("StridedSlice")
                    .CompileTimeConstantInput("begin")
                    .CompileTimeConstantInput("end")
                    .CompileTimeConstantInput("strides"),
                StridedSliceOp);

class StridedSliceGradOp : public XlaOpKernel {
 public:
  explicit StridedSliceGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("begin_mask", &begin_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("end_mask", &end_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ellipsis_mask", &ellipsis_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("new_axis_mask", &new_axis_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shrink_axis_mask", &shrink_axis_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Index", &index_type_));
  }

  // When the begin / end is unknown, compile the gradient into dynamic update
  // slice into a broadcasted 0s.
  //
  //    Broadcasted 0
  // +----------------------+
  // |         +----+       |
  // |<-begin->|grad|<-end->| <== Dynamic update grad into 0s.
  // |         +----+       |
  // +----------------------+
  void CompileAsDynamicUpdateSlice(XlaOpKernelContext* ctx,
                                   const TensorShape& input_shape,
                                   const xla::Literal& strides_literal) {
    bool dummy = false;
    Tensor strides_tensor;
    PartialTensorShape processing_shape, final_shape;
    absl::InlinedVector<int64_t, 4> begin;
    absl::InlinedVector<int64_t, 4> end;
    absl::InlinedVector<int64_t, 4> strides;
    StridedSliceShapeSpec shape_spec;
    OP_REQUIRES_OK(ctx, LiteralToHostTensor(strides_literal, index_type_,
                                            &strides_tensor));
    OP_REQUIRES_OK(
        ctx, ValidateStridedSliceOp(
                 nullptr, nullptr, strides_tensor, input_shape, begin_mask_,
                 end_mask_, ellipsis_mask_, new_axis_mask_, shrink_axis_mask_,
                 &processing_shape, &final_shape, &dummy, &dummy, &dummy,
                 &begin, &end, &strides, &shape_spec));
    for (int64_t i = 0; i < processing_shape.dims(); ++i) {
      OP_REQUIRES(
          ctx, strides[i] == 1,
          errors::InvalidArgument("Strides in strided slice grad have to be "
                                  "one when inputs are not constant."));
    }

    xla::XlaOp grad = ctx->Input(4);
    xla::Shape grad_shape = ctx->InputXlaShape(4).value();
    VLOG(1) << "xla grad shape" << grad_shape;
    VLOG(1) << "xla final_shape" << final_shape;
    VLOG(1) << "input_shape" << input_shape.DebugString();
    auto input_sizes = input_shape.dim_sizes();
    // For unknown output dim the bound of the output shape is input.  Pad and
    // double the size of input shape to leave enough buffer to avoid OOB
    // dynamic update slice.
    auto input_sizes_padded = input_shape.dim_sizes();
    bool need_padding = false;
    for (int64_t i = 0; i < processing_shape.dims(); ++i) {
      if (processing_shape.dim_size(i) == -1) {
        input_sizes_padded[i] *= 2;
        need_padding = true;
      }
    }
    for (int64_t i = 0; i < grad_shape.rank(); ++i) {
      // Use grad shape, which is known, to update unknown processing shape.
      // Grad shape is the output of the ValidateStridedSliceOp function in
      // forward pass, thus we use output_to_processing_mapping.
      if (shape_spec.output_to_processing_mapping[i] != -1) {
        processing_shape.set_dim(shape_spec.output_to_processing_mapping[i],
                                 grad_shape.dimensions(i));
      }
    }

    std::vector<xla::XlaOp> begins;
    begins.reserve(processing_shape.dims());
    for (int64_t i = 0; i < input_shape.dims(); ++i) {
      bool begin_mask = (1 << i) & shape_spec.begin_dense_mask;
      // Similarly, use processing_to_sparse_mapping to find out corresponding
      // begin dim of the gradient, as indices for dynamic update slice.
      int64_t begin_dim = shape_spec.processing_to_sparse_mapping[i];
      xla::XlaOp begin_index;
      auto zero = xla::Zero(ctx->builder(), ctx->InputXlaType("begin"));
      if (begin_mask) {
        begin_index = zero;
      } else {
        xla::XlaOp dim_size = xla::Slice(ctx->Input(0), {i}, {i + 1}, {1});
        dim_size = xla::Reshape(dim_size, {});
        begin_index =
            xla::Slice(ctx->Input(1), {begin_dim}, {begin_dim + 1}, {1});
        begin_index = xla::Reshape(begin_index, {});
        auto index_negative = xla::Lt(begin_index, zero);
        auto wrapped_index = xla::Add(dim_size, begin_index);
        // Wrap negative indices around.
        begin_index = xla::Select(index_negative, wrapped_index, begin_index);
      }
      begins.push_back(begin_index);
    }
    auto zero = XlaHelpers::Zero(ctx->builder(), ctx->expected_output_dtype(0));

    zero = xla::Broadcast(zero, input_sizes_padded);
    grad = xla::Reshape(grad, processing_shape.dim_sizes());
    grad = xla::DynamicUpdateSlice(zero, grad, begins);
    if (need_padding) {
      // We padded the input shape to avoid OOB when DUS. Now slice out the
      // padding in the final result.
      std::vector<int64_t> strides(input_shape.dims(), 1);
      std::vector<int64_t> start_indices(input_shape.dims(), 0);
      grad = xla::Slice(grad, start_indices, input_sizes, strides);
    }
    ctx->SetOutput(0, grad);
  }
  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape processing_shape, final_shape;
    absl::InlinedVector<int64_t, 4> begin;
    absl::InlinedVector<int64_t, 4> end;
    absl::InlinedVector<int64_t, 4> strides;

    TensorShape input_shape;
    OP_REQUIRES_OK(
        ctx, ctx->ConstantInputAsShape(0, &input_shape,
                                       xla::ValueInferenceMode::kUpperBound));
    xla::Literal begin_literal, end_literal, strides_literal;

    bool begin_is_constant = ctx->ConstantInput(1, &begin_literal).ok();
    bool end_is_constant = ctx->ConstantInput(2, &end_literal).ok();
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(3, &strides_literal));
    if (!(begin_is_constant && end_is_constant)) {
      CompileAsDynamicUpdateSlice(ctx, input_shape, strides_literal);
      return;
    }
    Tensor begin_tensor, end_tensor, strides_tensor;
    OP_REQUIRES_OK(
        ctx, LiteralToHostTensor(begin_literal, index_type_, &begin_tensor));
    OP_REQUIRES_OK(ctx,
                   LiteralToHostTensor(end_literal, index_type_, &end_tensor));
    OP_REQUIRES_OK(ctx, LiteralToHostTensor(strides_literal, index_type_,
                                            &strides_tensor));

    bool dummy = false;
    OP_REQUIRES_OK(
        ctx, ValidateStridedSliceOp(
                 &begin_tensor, &end_tensor, strides_tensor, input_shape,
                 begin_mask_, end_mask_, ellipsis_mask_, new_axis_mask_,
                 shrink_axis_mask_, &processing_shape, &final_shape, &dummy,
                 &dummy, &dummy, &begin, &end, &strides));

    // Check to make sure dy is consistent with the original slice
    const TensorShape dy_shape = ctx->InputShape(4);
    OP_REQUIRES(
        ctx, final_shape == dy_shape,
        errors::InvalidArgument("shape of dy was ", dy_shape.DebugString(),
                                " instead of ", final_shape.DebugString()));

    OP_REQUIRES(
        ctx, input_shape.dims() == processing_shape.dims(),
        errors::Internal(
            "input shape and processing shape must have same number of dims"));

    auto zero = XlaHelpers::Zero(ctx->builder(), ctx->expected_output_dtype(0));

    xla::XlaOp grad = ctx->Input(4);

    // Undo any new/shrink axes.
    grad = xla::Reshape(grad, processing_shape.dim_sizes());

    // Pad the input gradients.
    absl::InlinedVector<int64_t, 4> dimensions_to_reverse;
    xla::PaddingConfig padding_config;

    for (int i = 0; i < processing_shape.dims(); ++i) {
      auto* dims = padding_config.add_dimensions();
      if (strides[i] > 0) {
        dims->set_edge_padding_low(begin[i]);
        dims->set_interior_padding(strides[i] - 1);

        // Pad the upper dimension up to the expected input shape. (It's
        // not sufficient simply to use "end[i]" to compute the padding in
        // cases where the stride does not divide evenly into the interval
        // between begin[i] and end[i].)
        int64_t size =
            dims->edge_padding_low() + processing_shape.dim_size(i) +
            (processing_shape.dim_size(i) - 1) * dims->interior_padding();
        dims->set_edge_padding_high(input_shape.dim_size(i) - size);
      } else {
        dimensions_to_reverse.push_back(i);
        dims->set_edge_padding_high(input_shape.dim_size(i) - begin[i] - 1);
        dims->set_interior_padding(-strides[i] - 1);

        // Pad the lower dimension up to the expected input shape.
        int64_t size =
            dims->edge_padding_high() + processing_shape.dim_size(i) +
            (processing_shape.dim_size(i) - 1) * dims->interior_padding();
        dims->set_edge_padding_low(input_shape.dim_size(i) - size);
      }
    }
    if (!dimensions_to_reverse.empty()) {
      grad = xla::Rev(grad, dimensions_to_reverse);
    }
    grad = xla::Pad(grad, zero, padding_config);

    xla::XlaOp dynamic_shape = ctx->Input(0);
    xla::Shape grad_shape = ctx->builder()->GetShape(grad).value();
    std::vector<bool> dynamic_input;
    OP_REQUIRES_OK(ctx,
                   ctx->ResolveInputDynamismIntoPredVector(0, &dynamic_input));
    // Input of strided_slice_op has to have the same shape as output.
    DCHECK_EQ(grad_shape.rank(), input_shape.dims());
    for (int64_t dim = 0; dim < input_shape.dims(); ++dim) {
      DCHECK_EQ(grad_shape.dimensions(dim), input_shape.dim_size(dim));
      if (dynamic_input[dim]) {
        // Input is a dynamic dimension, set the same dynamic dimension size in
        // the output.
        auto dim_size = xla::Slice(dynamic_shape, {dim}, {dim + 1}, {1});
        dim_size = xla::ConvertElementType(dim_size, xla::S32);
        auto dim_size_scalar = xla::Reshape(dim_size, {});
        grad = xla::SetDimensionSize(grad, dim_size_scalar, dim);
      } else if (grad_shape.is_dynamic_dimension(dim)) {
        // Input is static but output is dynamic, respect input and remove any
        // dynamic dim in the output.
        grad = xla::RemoveDynamicDimension(grad, dim);
      }
    }

    ctx->SetOutput(0, grad);
  }

 private:
  int32 begin_mask_, end_mask_;
  int32 ellipsis_mask_, new_axis_mask_, shrink_axis_mask_;
  DataType index_type_;
};

REGISTER_XLA_OP(Name("StridedSliceGrad")
                    .CompileTimeConstantInput("shape")
                    .CompileTimeConstantInput("begin")
                    .CompileTimeConstantInput("end")
                    .CompileTimeConstantInput("strides"),
                StridedSliceGradOp);

class StridedSliceAssignOp : public XlaOpKernel {
 public:
  explicit StridedSliceAssignOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("begin_mask", &begin_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("end_mask", &end_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ellipsis_mask", &ellipsis_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("new_axis_mask", &new_axis_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shrink_axis_mask", &shrink_axis_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Index", &index_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape final_shape;
    absl::InlinedVector<int64_t, 4> begin;
    absl::InlinedVector<int64_t, 4> end;
    absl::InlinedVector<int64_t, 4> strides;

    xla::Literal begin_literal, end_literal, strides_literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(1, &begin_literal));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(2, &end_literal));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(3, &strides_literal));

    Tensor begin_tensor, end_tensor, strides_tensor;
    OP_REQUIRES_OK(
        ctx, LiteralToHostTensor(begin_literal, index_type_, &begin_tensor));
    OP_REQUIRES_OK(ctx,
                   LiteralToHostTensor(end_literal, index_type_, &end_tensor));
    OP_REQUIRES_OK(ctx, LiteralToHostTensor(strides_literal, index_type_,
                                            &strides_tensor));

    TensorShape lhs_shape;
    xla::XlaOp lhs;
    if (ctx->input_type(0) == DT_RESOURCE) {
      OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype_, &lhs_shape, &lhs));
    } else {
      lhs_shape = ctx->InputShape(0);
      lhs = ctx->Input(0);
    }

    const TensorShape rhs_shape = ctx->InputShape(4);

    TensorShape dummy_processing_shape;
    bool dummy = false;
    OP_REQUIRES_OK(ctx,
                   ValidateStridedSliceOp(
                       &begin_tensor, &end_tensor, strides_tensor, lhs_shape,
                       begin_mask_, end_mask_, ellipsis_mask_, new_axis_mask_,
                       shrink_axis_mask_, &dummy_processing_shape, &final_shape,
                       &dummy, &dummy, &dummy, &begin, &end, &strides));

    if (final_shape.num_elements() == 0 && rhs_shape.num_elements() == 0) {
      // DynamicUpdateSlice does not allow 0-element updates. We should probably
      // check that rhs_shape can be broadcast to final_shape, but that is
      // probably better handled when implementing broadcasting more generally.
      return;
    }

    // TODO(aselle): This check is too strong, we only should need
    // input_shape to be broadcastable to final_shape
    OP_REQUIRES(ctx, final_shape == rhs_shape,
                errors::Unimplemented(
                    "sliced l-value shape ", final_shape.DebugString(),
                    " does not match r-value shape ", rhs_shape.DebugString(),
                    ". Automatic broadcasting not yet implemented."));

    xla::XlaOp rhs = ctx->Input(4);

    absl::InlinedVector<int64_t, 4> dimensions_to_reverse;
    absl::InlinedVector<xla::XlaOp, 4> slice_begin;
    absl::InlinedVector<int64_t, 4> slice_dims;
    for (int i = 0; i < begin.size(); ++i) {
      // TODO(b/121179231): implement strides != 1
      OP_REQUIRES(
          ctx, strides[i] == 1 || strides[i] == -1,
          errors::Unimplemented("Strides != 1 or -1 are not yet implemented"));
      if (strides[i] > 0) {
        slice_begin.push_back(
            xla::ConstantR0<int64_t>(ctx->builder(), begin[i]));
        slice_dims.push_back(end[i] - begin[i]);
      } else {
        // Negative stride: swap begin and end, add 1 because the interval
        // is semi-open, and mark the dimension to be reversed.
        slice_begin.push_back(
            xla::ConstantR0<int64_t>(ctx->builder(), end[i] + 1));
        slice_dims.push_back(begin[i] - end[i]);
        dimensions_to_reverse.push_back(i);
      }
    }

    if (!dimensions_to_reverse.empty()) {
      rhs = xla::Rev(rhs, dimensions_to_reverse);
    }
    rhs = xla::Reshape(rhs, slice_dims);

    lhs = xla::DynamicUpdateSlice(lhs, rhs, slice_begin);

    if (ctx->input_type(0) == DT_RESOURCE) {
      OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, lhs));
    } else {
      ctx->SetOutput(0, lhs);
    }
  }

 private:
  int32 begin_mask_, end_mask_;
  int32 ellipsis_mask_, new_axis_mask_, shrink_axis_mask_;
  DataType index_type_;
  DataType dtype_;
};

REGISTER_XLA_OP(Name("ResourceStridedSliceAssign")
                    .CompileTimeConstantInput("begin")
                    .CompileTimeConstantInput("end")
                    .CompileTimeConstantInput("strides"),
                StridedSliceAssignOp);

REGISTER_XLA_OP(Name("TensorStridedSliceUpdate")
                    .CompileTimeConstantInput("begin")
                    .CompileTimeConstantInput("end")
                    .CompileTimeConstantInput("strides"),
                StridedSliceAssignOp);

}  // namespace
}  // namespace tensorflow
