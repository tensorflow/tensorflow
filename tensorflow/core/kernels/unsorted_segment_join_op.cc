/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/string_ops.cc.

#include <string>
#include <utility>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

namespace {

template <typename INDICES_TYPE>
gtl::InlinedVector<INDICES_TYPE, 8> GetFlattenedRelativeOffsets(
    INDICES_TYPE small_stride, INDICES_TYPE big_stride) {
  gtl::InlinedVector<INDICES_TYPE, 8> flattened_offsets(small_stride);
  for (auto i = 0; i < small_stride; i++) {
    flattened_offsets[i] = i * big_stride;
  }
  return flattened_offsets;
}

template <typename INDICES_TYPE>
std::pair<INDICES_TYPE, INDICES_TYPE> GetStrides(
    const TensorShape& input_shape, const TensorShape& segment_id_shape) {
  int64 small_stride = 1;
  int64 big_stride = 1;
  for (auto i = 0; i < input_shape.dims(); i++) {
    if (i < segment_id_shape.dims()) {
      small_stride *= segment_id_shape.dim_size(i);
    } else {
      big_stride *= input_shape.dim_size(i);
    }
  }
  return std::make_pair(big_stride, small_stride);
}

TensorShape GetOutputShape(const TensorShape& input_shape,
                           const TensorShape& segment_id_shape,
                           const int64 num_segments) {
  TensorShape output_shape;
  output_shape.AddDim(num_segments);
  for (size_t index = segment_id_shape.dims(); index < input_shape.dims();
       ++index) {
    output_shape.AddDim(input_shape.dim_size(index));
  }
  return output_shape;
}

}  // namespace

template <typename INDICES_TYPE, typename NUM_SEGMENTS_TYPE>
class UnsortedSegmentJoinOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit UnsortedSegmentJoinOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("separator", &separator_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    const int32 input_dims = input_shape.dims();

    const Tensor& segment_id = context->input(1);
    const TensorShape& segment_id_shape = segment_id.shape();
    const int32 segment_dims = segment_id_shape.dims();

    const Tensor& num_segments_tensor = context->input(2);
    OP_REQUIRES(context, num_segments_tensor.NumElements() != 0,
                errors::InvalidArgument("Number of segments cannot be empty."));
    auto num_segments = num_segments_tensor.scalar<NUM_SEGMENTS_TYPE>()();

    OP_REQUIRES(context, segment_dims != 0,
                errors::InvalidArgument("Segment_id cannot have rank 0"));

    OP_REQUIRES(
        context, segment_dims <= input_dims,
        errors::OutOfRange("Invalid segment_id rank ", segment_dims,
                           " for input with ", input_dims, " dimension(s)"));
    for (auto i = 0; i < segment_dims; i++) {
      OP_REQUIRES(
          context, segment_id_shape.dim_size(i) == input_shape.dim_size(i),
          errors::InvalidArgument(
              "Segment dimension is ", segment_id_shape.dim_size(i),
              " while input dimension is ", input_dims, " in rank ", i));
    }

    // Making output tensor.
    Tensor* output_tensor = nullptr;
    TensorShape output_shape =
        GetOutputShape(input_shape, segment_id_shape, num_segments);
    OP_REQUIRES_OK(context, context->allocate_output("output", output_shape,
                                                     &output_tensor));

    // Preparating flat tensors.
    auto output_flat = output_tensor->flat<tstring>();
    auto flat_segment_id = segment_id.flat<INDICES_TYPE>();
    auto flat_input = input.flat<tstring>();

    for (int i = 0; i < flat_segment_id.size(); i++) {
      OP_REQUIRES(
          context,
          ((flat_segment_id(i) < num_segments) && (flat_segment_id(i) >= 0)),
          errors::InvalidArgument(
              "segment_ids are not allowed to exceed num_segments or"
              " to have negative values."));
    }

    int64 big_stride;
    int64 small_stride;
    std::tie(big_stride, small_stride) =
        GetStrides<INDICES_TYPE>(input_shape, segment_id_shape);
    auto relative_offset_set =
        GetFlattenedRelativeOffsets<INDICES_TYPE>(small_stride, big_stride);
    for (auto start_offset = 0; start_offset < big_stride; start_offset++) {
      for (auto i = 0; i < relative_offset_set.size(); i++) {
        auto output_index = start_offset + flat_segment_id(i) * big_stride;
        auto offset = start_offset + relative_offset_set[i];
        if (output_flat(output_index).length() != 0)
          output_flat(output_index).append(separator_.c_str());
        output_flat(output_index).append(flat_input(offset));
      }
    }
  }

 private:
  string separator_;
};

#define REGISTER_CPU_KERNEL(indices_type, num_segments_type)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("UnsortedSegmentJoin")                             \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<indices_type>("Tindices")           \
          .TypeConstraint<num_segments_type>("Tnumsegments"), \
      UnsortedSegmentJoinOp<indices_type, num_segments_type>);

REGISTER_CPU_KERNEL(int32, int32);
REGISTER_CPU_KERNEL(int32, int64);
REGISTER_CPU_KERNEL(int64, int32);
REGISTER_CPU_KERNEL(int64, int64);
#undef REGISTER_CPU_KERNEL

}  // namespace tensorflow
