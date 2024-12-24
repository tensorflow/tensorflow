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

#include <cstdint>
#include <vector>

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/value_inference.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

class SegmentReduce : public XlaOpKernel {
 public:
  explicit SegmentReduce(OpKernelConstruction* ctx, bool indices_are_sorted)
      : XlaOpKernel(ctx), indices_are_sorted_(indices_are_sorted) {
    DataType dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &type_));
  }

  // The initial value to initialize elements of the output to.
  virtual xla::XlaOp InitialValue(xla::XlaBuilder* builder) = 0;

  // A function to combine two scalars with the same index (e.g., sum).
  virtual xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) = 0;

  void Compile(XlaOpKernelContext* ctx) override {
    // output = unsorted_segment_sum(data, indices, num_segments)
    // Compute a tensor such that:
    //    output[i] = sum over {j where indices[j] == i} of data[j]
    //    output[i] == 0 if i does not appear in indices
    //
    // Contrast with segment_sum(), which assumes indices are sorted and that
    // max(indices)+1 is the desired size of the output. Note that
    // segment_sum_v2 also takes num_segments as an input and can be supported
    // similarly.
    //
    // The returned output tensor has the same type as data, and the same shape
    // as data with the first indices.rank dimensions are replaced
    // by a single dimension with size num_segments.
    auto data = ctx->Input(0);
    TensorShape data_shape = ctx->InputShape(0);

    auto indices = ctx->Input(1);
    TensorShape indices_shape = ctx->InputShape(1);

    int64_t num_segments;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntScalar(
                       2, &num_segments, xla::ValueInferenceMode::kUpperBound));
    OP_REQUIRES(ctx, data_shape.dims() >= indices_shape.dims(),
                errors::InvalidArgument(type_string(),
                                        " requires that indices' rank be"
                                        " less than or equal to data's rank."));
    // Validate that indices.shape is a prefix of data.shape.
    for (int d = 0; d < indices_shape.dims(); ++d) {
      OP_REQUIRES(
          ctx, (data_shape.dim_size(d) == indices_shape.dim_size(d)),
          errors::InvalidArgument(type_string(),
                                  " requires indices shape to be prefix"
                                  " of data_shape, but dimension ",
                                  d, " differs ", data_shape.dim_size(d),
                                  " vs. ", indices_shape.dim_size(d)));
    }
    xla::XlaBuilder* builder = ctx->builder();
    // data shape = [indices_shape, segment_shape]
    // buffer shape = [num_segment, segment_shape]
    // We now create the buffer shape by reverse enginerring data shape into
    // indices shape and segment shape.
    TensorShape buffer_shape = data_shape;
    buffer_shape.RemoveDimRange(0, indices_shape.dims());
    buffer_shape.InsertDim(0, num_segments);

    auto buffer =
        xla::Broadcast(InitialValue(builder), buffer_shape.dim_sizes());

    // Build dynamic dim sizes for buffer, as well as whether each dimension
    // size is dynamic or static. We build two parts: num_sgement part and
    // segment_shape part.
    std::vector<xla::XlaOp> buffer_dims;
    std::vector<bool> buffer_dims_are_dynamic;
    // Build the "num_segment" part.
    bool num_segments_is_dynamic;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPred(2, &num_segments_is_dynamic));

    buffer_dims.insert(buffer_dims.begin(), ctx->Input(2));
    buffer_dims_are_dynamic.insert(buffer_dims_are_dynamic.begin(),
                                   num_segments_is_dynamic);
    // Build the segment shape part.
    for (int64_t i = indices_shape.dims(); i < data_shape.dims(); ++i) {
      buffer_dims.push_back(xla::GetDimensionSize(data, i));
      buffer_dims_are_dynamic.push_back(
          ctx->InputXlaShape(0)->is_dynamic_dimension(i));
    }

    for (int64_t i = 0; i < buffer_dims.size(); ++i) {
      if (buffer_dims_are_dynamic[i]) {
        // For each dynamic dimension, call set-dimension-size on it.
        buffer = xla::SetDimensionSize(buffer, buffer_dims[i], i);
      }
    }

    auto combiner = [this](xla::XlaOp a, xla::XlaOp b,
                           xla::XlaBuilder* builder) { return Combine(a, b); };

    auto result = XlaScatter(buffer, /*updates=*/data, indices,
                             /*indices_are_vectors=*/false, indices_are_sorted_,
                             combiner, builder);
    OP_REQUIRES_OK(ctx, result.status());
    ctx->SetOutput(0, result.value());
  }

 protected:
  xla::PrimitiveType type_;
  bool indices_are_sorted_;
};

template <bool indices_are_sorted>
class SegmentSum : public SegmentReduce {
 public:
  explicit SegmentSum(OpKernelConstruction* ctx)
      : SegmentReduce(ctx, indices_are_sorted) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::Zero(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override { return a + b; };
};

REGISTER_XLA_OP(Name("SegmentSumV2").CompileTimeConstantInput("num_segments"),
                SegmentSum</*indices_are_sorted=*/true>);
REGISTER_XLA_OP(
    Name("UnsortedSegmentSum").CompileTimeConstantInput("num_segments"),
    SegmentSum</*indices_are_sorted=*/false>);

template <bool indices_are_sorted>
class SegmentProd : public SegmentReduce {
 public:
  explicit SegmentProd(OpKernelConstruction* ctx)
      : SegmentReduce(ctx, indices_are_sorted) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::One(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override { return a * b; };
};

REGISTER_XLA_OP(
    Name("UnsortedSegmentProd").CompileTimeConstantInput("num_segments"),
    SegmentProd</*indices_are_sorted=*/false>);
REGISTER_XLA_OP(Name("SegmentProdV2").CompileTimeConstantInput("num_segments"),
                SegmentProd</*indices_are_sorted=*/true>);

template <bool indices_are_sorted>
class SegmentMin : public SegmentReduce {
 public:
  explicit SegmentMin(OpKernelConstruction* ctx)
      : SegmentReduce(ctx, indices_are_sorted) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::MaxFiniteValue(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override {
    return xla::Min(a, b);
  };
};

REGISTER_XLA_OP(
    Name("UnsortedSegmentMin").CompileTimeConstantInput("num_segments"),
    SegmentMin</*indices_are_sorted=*/false>);
REGISTER_XLA_OP(Name("SegmentMinV2").CompileTimeConstantInput("num_segments"),
                SegmentMin</*indices_are_sorted=*/true>);

template <bool indices_are_sorted>
class SegmentMax : public SegmentReduce {
 public:
  explicit SegmentMax(OpKernelConstruction* ctx)
      : SegmentReduce(ctx, indices_are_sorted) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::MinFiniteValue(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override {
    return xla::Max(a, b);
  };
};

REGISTER_XLA_OP(
    Name("UnsortedSegmentMax").CompileTimeConstantInput("num_segments"),
    SegmentMax</*indices_are_sorted=*/false>);
REGISTER_XLA_OP(Name("SegmentMaxV2").CompileTimeConstantInput("num_segments"),
                SegmentMax</*indices_are_sorted=*/true>);

}  // namespace
}  // namespace tensorflow
