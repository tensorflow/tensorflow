/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class ReverseSequenceOp : public XlaOpKernel {
 public:
  explicit ReverseSequenceOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("batch_dim", &batch_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("seq_dim", &seq_dim_));
  }

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    const TensorShape seq_lens_shape = context->InputShape(1);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(seq_lens_shape),
                errors::InvalidArgument("seq_lengths must be 1-dim, not ",
                                        seq_lens_shape.dims()));
    OP_REQUIRES(context, batch_dim_ != seq_dim_,
                errors::InvalidArgument("batch_dim == seq_dim == ", seq_dim_));
    OP_REQUIRES(context, seq_dim_ >= 0,
                errors::InvalidArgument("seq_dim must be >=0, got ", seq_dim_));
    OP_REQUIRES(
        context, seq_dim_ < input_shape.dims(),
        errors::InvalidArgument("seq_dim must be < input rank", " ( ", seq_dim_,
                                " vs. ", input_shape.dims(), ")"));
    OP_REQUIRES(
        context, batch_dim_ >= 0,
        errors::InvalidArgument("batch_dim must be >=0, got ", batch_dim_));
    OP_REQUIRES(
        context, batch_dim_ < input_shape.dims(),
        errors::InvalidArgument("batch_dim must be < input rank", " ( ",
                                batch_dim_, " vs. ", input_shape.dims(), ")"));
    OP_REQUIRES(
        context,
        seq_lens_shape.num_elements() == input_shape.dim_size(batch_dim_),
        errors::InvalidArgument("Length of seq_lengths != input.dims(",
                                batch_dim_, "), ", "(",
                                seq_lens_shape.num_elements(), " vs. ",
                                input_shape.dim_size(batch_dim_), ")"));

    xla::XlaBuilder* builder = context->builder();
    const auto input = context->Input(0);
    const auto seq_lens = context->Input(1);

    const int64_t batch_size = input_shape.dim_size(batch_dim_);
    if (batch_size == 0) {
      context->SetOutput(0, input);
      return;
    }

    const xla::PrimitiveType seq_lens_type = context->input_xla_type(1);
    const int64_t max_seq_len = input_shape.dim_size(seq_dim_);

    // Create [batch, sequence, 2] tensor that contains the indices where the
    // real data belongs
    xla::XlaOp back = xla::Sub(seq_lens, xla::ScalarLike(seq_lens, 1));
    xla::XlaOp batch_idx = xla::Iota(
        builder,
        xla::ShapeUtil::MakeShape(seq_lens_type, {batch_size, max_seq_len, 1}),
        /*iota_dimension=*/0);
    xla::XlaOp forward_idx = xla::Iota(
        builder,
        xla::ShapeUtil::MakeShape(seq_lens_type, {batch_size, max_seq_len, 1}),
        /*iota_dimension=*/1);
    xla::XlaOp reverse_idx = xla::Sub(back, forward_idx, {0});
    reverse_idx = xla::Select(xla::Lt(reverse_idx, xla::ZerosLike(reverse_idx)),
                              forward_idx, reverse_idx);
    if (batch_dim_ > seq_dim_) {
      // The output of the XLA gather op keeps indices dimensions in the same
      // order as they appear in the input. If the batch_dim_ needs to be after
      // the seq_dim_ in the output, it also needs to be that way in the input
      // so we transpose.
      batch_idx = xla::Transpose(batch_idx, {1, 0, 2});
      forward_idx = xla::Transpose(forward_idx, {1, 0, 2});
      reverse_idx = xla::Transpose(reverse_idx, {1, 0, 2});
    }
    xla::XlaOp start_indices =
        xla::ConcatInDim(builder, {batch_idx, reverse_idx},
                         /*dimension=*/2);

    xla::GatherDimensionNumbers dnums;
    dnums.set_index_vector_dim(2);
    // The first and second element in the third dimension of reverse_idx are
    // the batch_dim_ offset and the seq_dim_ offset respectively.
    dnums.add_start_index_map(batch_dim_);
    dnums.add_start_index_map(seq_dim_);

    // batch_dim_ and seq_dim_ are collapsed and the other dimensions are kept
    // in the gather.
    for (int i = 0; i < input_shape.dims(); ++i) {
      if (i != batch_dim_ && i != seq_dim_) {
        dnums.add_offset_dims(i);
      } else {
        dnums.add_collapsed_slice_dims(i);
      }
    }

    auto slice_sizes = input_shape.dim_sizes();
    slice_sizes[batch_dim_] = 1;
    slice_sizes[seq_dim_] = 1;

    context->SetOutput(0,
                       xla::Gather(input, start_indices, dnums, slice_sizes));
  }

 private:
  int32 batch_dim_;
  int32 seq_dim_;
};

REGISTER_XLA_OP(Name("ReverseSequence"), ReverseSequenceOp);

}  // namespace
}  // namespace tensorflow
