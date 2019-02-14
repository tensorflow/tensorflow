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
                errors::InvalidArgument("seq_lens input must be 1-dim, not ",
                                        seq_lens_shape.dims()));
    OP_REQUIRES(context, batch_dim_ != seq_dim_,
                errors::InvalidArgument("batch_dim == seq_dim == ", seq_dim_));
    OP_REQUIRES(
        context, seq_dim_ < input_shape.dims(),
        errors::InvalidArgument("seq_dim must be < input.dims()", "( ",
                                seq_dim_, " vs. ", input_shape.dims(), ")"));
    OP_REQUIRES(
        context, batch_dim_ < input_shape.dims(),
        errors::InvalidArgument("batch_dim must be < input.dims()", "( ",
                                batch_dim_, " vs. ", input_shape.dims(), ")"));
    OP_REQUIRES(
        context,
        seq_lens_shape.num_elements() == input_shape.dim_size(batch_dim_),
        errors::InvalidArgument("len(seq_lens) != input.dims(", batch_dim_,
                                "), ", "(", seq_lens_shape.num_elements(),
                                " vs. ", input_shape.dim_size(batch_dim_)));

    xla::XlaBuilder* builder = context->builder();
    const auto input = context->Input(0);
    const auto seq_lens = context->Input(1);

    const int64 batch_size = input_shape.dim_size(batch_dim_);
    if (batch_size == 0) {
      context->SetOutput(0, input);
      return;
    }

    // Given the input
    //
    // 012345
    // 6789AB
    //
    // and sequence lens {2, 3} we:
    //
    // 1. Reverse and pad each row to get
    //
    //    543210XXXXXX
    //    BA9876XXXXXX
    //
    // 2. Gather out the suffix from each row to get
    //
    //    10XXXX
    //    876XXX
    //
    // 3. Select from the input and the array created by (2) to get the result.
    //
    //    102345
    //    8769AB
    const xla::PrimitiveType input_type = context->input_xla_type(0);
    const xla::PrimitiveType seq_lens_type = context->input_xla_type(1);
    const int64 max_seq_len = input_shape.dim_size(seq_dim_);

    xla::XlaOp rev = xla::Rev(input, {seq_dim_});

    auto padding_config = xla::MakeNoPaddingConfig(input_shape.dims());
    padding_config.mutable_dimensions(seq_dim_)->set_edge_padding_high(
        max_seq_len);
    xla::XlaOp padded =
        xla::Pad(rev, xla::Zero(builder, input_type), padding_config);

    // Form a start indices tensor with shape [2, batch_size]. For each batch
    // entry we have a (batch offset, seq offset) pair.
    xla::XlaOp start_indices = xla::ConcatInDim(
        builder,
        {
            xla::Iota(builder,
                      xla::ShapeUtil::MakeShape(seq_lens_type, {1, batch_size}),
                      /*iota_dimension=*/1),
            xla::Reshape(xla::ScalarLike(seq_lens, max_seq_len) - seq_lens,
                         {1, batch_size}),
        },
        /*dimension=*/0);

    xla::GatherDimensionNumbers dnums;
    // The first dimension of start_indices contains the batch/seq dim choice.
    dnums.set_index_vector_dim(0);
    dnums.add_start_index_map(batch_dim_);
    dnums.add_start_index_map(seq_dim_);

    // All other dimensions other than the batch dim are offset dimensions.
    for (int i = 0; i < input_shape.dims(); ++i) {
      if (i != batch_dim_) {
        dnums.add_offset_dims(i);
      }
    }
    dnums.add_collapsed_slice_dims(batch_dim_);

    auto slice_sizes = input_shape.dim_sizes();
    slice_sizes[batch_dim_] = 1;

    xla::XlaOp output = xla::Gather(padded, start_indices, dnums, slice_sizes);

    // Mask out elements after the sequence length, and copy the corresponding
    // elements from the input.
    xla::XlaOp iota = xla::Iota(builder, seq_lens_type, max_seq_len);
    std::vector<int64> dims(input_shape.dims(), 1);
    dims[batch_dim_] = batch_size;
    auto mask = xla::Lt(iota, xla::Reshape(seq_lens, dims), {seq_dim_});

    // Broadcast the mask up to the input shape.
    mask = xla::Or(mask, xla::Broadcast(xla::ConstantR0<bool>(builder, false),
                                        input_shape.dim_sizes()));

    output = xla::Select(mask, output, input);
    context->SetOutput(0, output);
  }

 private:
  int32 batch_dim_;
  int32 seq_dim_;
};

REGISTER_XLA_OP(Name("ReverseSequence"), ReverseSequenceOp);

}  // namespace
}  // namespace tensorflow
