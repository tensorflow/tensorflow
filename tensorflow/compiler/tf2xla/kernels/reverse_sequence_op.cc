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
#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
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

    const DataType input_type = context->input_type(0);
    const DataType seq_lens_type = context->input_type(1);
    const int64 max_seq_len = input_shape.dim_size(seq_dim_);

    xla::Shape input_xla_shape;
    OP_REQUIRES_OK(context, TensorShapeToXLAShape(input_type, input_shape,
                                                  &input_xla_shape));
    xla::Shape seq_lens_xla_shape;
    OP_REQUIRES_OK(context, TensorShapeToXLAShape(seq_lens_type, seq_lens_shape,
                                                  &seq_lens_xla_shape));

    const auto tuple_shape = xla::ShapeUtil::MakeTupleShape({
        xla::ShapeUtil::MakeShape(seq_lens_xla_shape.element_type(), {}),
        seq_lens_xla_shape,
        input_xla_shape,
    });

    // For each entry in the batch, reverse the sequence.
    // TODO(b/65689298): generalize the Map() operator to non-scalar cases and
    // use it here, instead of a While loop.

    // Condition: lambda (i, _, _): i < batch_size
    auto condition_builder =
        builder->CreateSubBuilder("reverse_sequence_condition");
    {
      auto param =
          xla::Parameter(condition_builder.get(), 0, tuple_shape, "param");
      auto i = xla::GetTupleElement(param, 0);
      xla::Lt(i, XlaHelpers::IntegerLiteral(condition_builder.get(),
                                            seq_lens_type, batch_size));
    }
    auto condition = condition_builder->Build();
    OP_REQUIRES_OK(context, condition.status());

    auto body_builder = builder->CreateSubBuilder("reverse_sequence_body");
    {
      auto param = xla::Parameter(body_builder.get(), 0, tuple_shape, "param");
      auto i = xla::GetTupleElement(param, 0);
      auto seq_lens = xla::GetTupleElement(param, 1);
      auto output = xla::GetTupleElement(param, 2);

      // seq_len is the sequence length of the current batch element (rank 1)
      auto seq_len = xla::DynamicSlice(seq_lens, xla::Reshape(i, {1}), {1});

      // Indices is the offset of the batch element in the input.
      auto batch_element_indices =
          xla::Broadcast(XlaHelpers::Zero(body_builder.get(), seq_lens_type),
                         {input_shape.dims()});
      batch_element_indices = xla::DynamicUpdateSlice(
          batch_element_indices, xla::Reshape(i, {1}),
          xla::Reshape(XlaHelpers::IntegerLiteral(body_builder.get(),
                                                  seq_lens_type, batch_dim_),
                       {1}));

      // Slice out the current batch element and pad it out in the sequence
      // dimension.
      TensorShape slice_shape = input_shape;
      slice_shape.set_dim(batch_dim_, 1);
      slice_shape.set_dim(seq_dim_, max_seq_len);
      auto slice = xla::DynamicSlice(output, batch_element_indices,
                                     slice_shape.dim_sizes());
      auto padding_config = xla::MakeNoPaddingConfig(slice_shape.dims());
      padding_config.mutable_dimensions(seq_dim_)->set_edge_padding_high(
          slice_shape.dim_size(seq_dim_));
      slice = xla::Pad(slice, XlaHelpers::Zero(body_builder.get(), input_type),
                       padding_config);

      // Now slice out the reversed sequence from its actual start.
      // sequence_start_indices is the offset of the start of the reversed
      // sequence in the input. The slice will go into the padding, however, we
      // will mask off these elements and replace them with elements from the
      // original input so their values do not matter.
      auto sequence_start_indices =
          xla::Broadcast(XlaHelpers::Zero(body_builder.get(), seq_lens_type),
                         {slice_shape.dims()});
      sequence_start_indices = xla::DynamicUpdateSlice(
          sequence_start_indices,
          xla::Sub(XlaHelpers::IntegerLiteral(body_builder.get(), seq_lens_type,
                                              max_seq_len),
                   seq_len),
          xla::Reshape(XlaHelpers::IntegerLiteral(body_builder.get(),
                                                  seq_lens_type, seq_dim_),
                       {1}));
      slice = xla::DynamicSlice(slice, sequence_start_indices,
                                slice_shape.dim_sizes());

      // Shift the reversed sequence to the left.
      output = xla::DynamicUpdateSlice(output, slice, batch_element_indices);

      xla::Tuple(
          body_builder.get(),
          {xla::Add(i, XlaHelpers::One(body_builder.get(), seq_lens_type)),
           seq_lens, output});
    }
    auto body = body_builder->Build();
    OP_REQUIRES_OK(context, body.status());

    auto loop_output = xla::While(
        condition.ValueOrDie(), body.ValueOrDie(),
        xla::Tuple(builder, {XlaHelpers::Zero(builder, seq_lens_type), seq_lens,
                             xla::Rev(input, {seq_dim_})}));
    auto output = xla::GetTupleElement(loop_output, 2);

    // Mask out elements after the sequence length.
    xla::XlaOp iota =
        xla::Iota(builder, seq_lens_xla_shape.element_type(), max_seq_len);
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
