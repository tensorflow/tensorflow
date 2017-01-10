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
#include "tensorflow/compiler/tf2xla/literal_util.h"
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

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);

    TensorShape final_shape;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

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

    TensorShape dummy_processing_shape;
    ShapeReadWriteFromTensorShape wrapped_final_shape(&final_shape);
    ShapeReadWriteFromTensorShape wrapped_dummy_processing_shape(
        &dummy_processing_shape);
    bool dummy = false;
    OP_REQUIRES_OK(
        ctx, ValidateStridedSliceOp(
                 &begin_tensor, &end_tensor, strides_tensor,
                 ShapeReadWriteFromTensorShape(&input_shape), begin_mask_,
                 end_mask_, ellipsis_mask_, new_axis_mask_, shrink_axis_mask_,
                 &wrapped_dummy_processing_shape, &wrapped_final_shape, &dummy,
                 &dummy, &dummy, &begin, &end, &strides));

    gtl::InlinedVector<int64, 4> dimensions_to_reverse;
    gtl::InlinedVector<int64, 4> slice_begin, slice_end;
    for (int i = 0; i < begin.size(); ++i) {
      // TODO(phawkins): implement strides != 1 when b/30878775 is fixed.
      OP_REQUIRES(
          ctx, strides[i] == 1 || strides[i] == -1,
          errors::Unimplemented("Strides != 1 or -1 are not yet implemented"));
      if (strides[i] > 0) {
        slice_begin.push_back(begin[i]);
        slice_end.push_back(end[i]);
      } else {
        // Negative stride: swap begin and end, add 1 because the interval
        // is semi-open, and mark the dimension to be reversed.
        slice_begin.push_back(end[i] + 1);
        slice_end.push_back(begin[i] + 1);
        dimensions_to_reverse.push_back(i);
      }
    }
    xla::ComputationDataHandle slice =
        ctx->builder()->Slice(ctx->Input(0), slice_begin, slice_end);
    if (!dimensions_to_reverse.empty()) {
      slice = ctx->builder()->Rev(slice, dimensions_to_reverse);
    }

    slice = ctx->builder()->Reshape(slice, final_shape.dim_sizes());
    ctx->SetOutput(0, slice);
  }

 private:
  int32 begin_mask_, end_mask_;
  int32 ellipsis_mask_, new_axis_mask_, shrink_axis_mask_;
  DataType index_type_;
};

REGISTER_XLA_OP("StridedSlice", StridedSliceOp);

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

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape processing_shape, final_shape;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    TensorShape input_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &input_shape));

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

    bool dummy = false;
    ShapeReadWriteFromTensorShape wrapped_final_shape(&final_shape);
    ShapeReadWriteFromTensorShape wrapped_processing_shape(&processing_shape);
    OP_REQUIRES_OK(
        ctx, ValidateStridedSliceOp(
                 &begin_tensor, &end_tensor, strides_tensor,
                 ShapeReadWriteFromTensorShape(&input_shape), begin_mask_,
                 end_mask_, ellipsis_mask_, new_axis_mask_, shrink_axis_mask_,
                 &wrapped_processing_shape, &wrapped_final_shape, &dummy,
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

    xla::ComputationDataHandle grad = ctx->Input(4);

    // Undo any new/shrink axes.
    grad = ctx->builder()->Reshape(grad, processing_shape.dim_sizes());

    // Pad the input gradients.
    gtl::InlinedVector<int64, 4> dimensions_to_reverse;
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
        int64 size =
            dims->edge_padding_low() + processing_shape.dim_size(i) +
            (processing_shape.dim_size(i) - 1) * dims->interior_padding();
        dims->set_edge_padding_high(input_shape.dim_size(i) - size);
      } else {
        dimensions_to_reverse.push_back(i);
        dims->set_edge_padding_high(input_shape.dim_size(i) - begin[i] - 1);
        dims->set_interior_padding(-strides[i] - 1);

        // Pad the lower dimension up to the expected input shape.
        int64 size =
            dims->edge_padding_high() + processing_shape.dim_size(i) +
            (processing_shape.dim_size(i) - 1) * dims->interior_padding();
        dims->set_edge_padding_low(input_shape.dim_size(i) - size);
      }
    }
    if (!dimensions_to_reverse.empty()) {
      grad = ctx->builder()->Rev(grad, dimensions_to_reverse);
    }
    grad = ctx->builder()->Pad(grad, zero, padding_config);
    ctx->SetOutput(0, grad);
  }

 private:
  int32 begin_mask_, end_mask_;
  int32 ellipsis_mask_, new_axis_mask_, shrink_axis_mask_;
  DataType index_type_;
};

REGISTER_XLA_OP("StridedSliceGrad", StridedSliceGradOp);

}  // namespace
}  // namespace tensorflow
