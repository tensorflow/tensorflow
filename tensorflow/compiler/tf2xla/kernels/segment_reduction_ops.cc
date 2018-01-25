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

#include <sstream>
#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

xla::ComputationDataHandle XlaComputeScatterAddDynamicSlice(
    XlaOpKernelContext* ctx, const xla::ComputationDataHandle& input,
    const TensorShape& input_shape, const xla::ComputationDataHandle& indices,
    const TensorShape& indices_shape, int64 num_segments, DataType dtype,
    xla::ComputationBuilder* builder) {
  // Flatten data for dynamic indexing via indices_1d.
  TensorShape input_shape_i(input_shape);
  for (int64 d = 0; d < indices_shape.dims(); ++d) {
    input_shape_i.RemoveDim(0);
  }
  TensorShape flat_shape({indices_shape.num_elements()});
  flat_shape.AppendShape(input_shape_i);

  // output is same as flattened input shape with dim_size(0) = num_segments.
  TensorShape out_shape(flat_shape);
  out_shape.set_dim(0, num_segments);

  // Slices from the input data are same shape as the input data, except dim 0.
  TensorShape slice_shape(flat_shape);
  slice_shape.set_dim(0, 1);
  TensorShape loop_out_slice_shape(out_shape);
  loop_out_slice_shape.set_dim(0, 1);

  // Construct the initial values of the loop-carried variables
  // Flatten the indices into 1-D for ease of iteration.
  auto indices_1d = builder->Reshape(indices, {indices_shape.num_elements()});
  // Flatten the data for ease of indexing via values in indices_1d.
  auto data_flat = builder->Reshape(input, flat_shape.dim_sizes());

  auto init_i = builder->ConstantR0<int32>(0);
  auto init_out = builder->Broadcast(XlaHelpers::Zero(builder, dtype),
                                     out_shape.dim_sizes());

  xla::PrimitiveType ptype;
  TF_CHECK_OK(DataTypeToPrimitiveType(dtype, &ptype));

  std::vector<xla::Shape> tuple_shapes(
      {// The loop iteration counter is a scalar, incremented each iteration.
       xla::ShapeUtil::MakeShape(xla::S32, {}),
       // The flattened input data is loop invariant.
       xla::ShapeUtil::MakeShape(ptype, flat_shape.dim_sizes()),
       // The scatter indices tensor is loop invariant.
       xla::ShapeUtil::MakeShape(xla::S32, {indices_shape.num_elements()}),
       // The output data array is updated each loop iteration.
       xla::ShapeUtil::MakeShape(ptype, out_shape.dim_sizes())});
  xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(tuple_shapes);

  auto init = builder->Tuple({init_i, data_flat, indices_1d, init_out});

  // Construct the while loop condition (i < num_indices)
  xla::ComputationBuilder condb(ctx->builder()->client(),
                                "ScatterAddWhileCond");
  condb.Lt(condb.GetTupleElement(
               condb.Parameter(0, tuple_shape, "ScatterAddWhileTuple"), 0),
           condb.ConstantR0<int32>(indices_shape.num_elements()));
  auto cond_status = condb.Build();
  auto cond = cond_status.ConsumeValueOrDie();

  // Construct the while loop body's function. The implementation of scatter is:
  // for i in range(num_indices):
  //   index = dynamic-slice(indices, i)
  //   xi = dynamic-slice(input, i)
  //   output = dynamic-update-slice(output, xi, index)
  xla::ComputationBuilder bodyb(ctx->builder()->client(),
                                "ScatterAddWhileBody");
  {
    auto input_tuple = bodyb.Parameter(0, tuple_shape, "ScatterAddWhileTuple");
    auto i = bodyb.GetTupleElement(input_tuple, 0);
    auto data = bodyb.GetTupleElement(input_tuple, 1);
    auto idcs = bodyb.GetTupleElement(input_tuple, 2);
    auto output = bodyb.GetTupleElement(input_tuple, 3);

    // Index into the data array at i.
    auto zero = bodyb.ConstantR1<int32>({0});
    std::vector<xla::ComputationDataHandle> index_vals(flat_shape.dims(), zero);
    index_vals[0] = bodyb.Reshape(i, {1});
    auto index = bodyb.ConcatInDim(index_vals, 0);

    auto data_slice =
        bodyb.Reshape(bodyb.DynamicSlice(data, index, slice_shape.dim_sizes()),
                      loop_out_slice_shape.dim_sizes());

    // Index into the output array.
    std::vector<xla::ComputationDataHandle> out_index_vals(out_shape.dims(),
                                                           zero);
    out_index_vals[0] = bodyb.DynamicSlice(idcs, bodyb.Reshape(i, {1}), {1});
    auto out_index = bodyb.ConcatInDim(out_index_vals, 0);

    // Slice the output array, update value, and update the output slice.
    auto updated_output = bodyb.DynamicUpdateSlice(
        output,
        bodyb.Add(data_slice,
                  bodyb.DynamicSlice(output, out_index,
                                     loop_out_slice_shape.dim_sizes())),
        out_index);

    auto ip1 = bodyb.Add(i, bodyb.ConstantR0<int32>(1));
    bodyb.Tuple({ip1, data, idcs, updated_output});
  }
  auto body_status = bodyb.Build();
  auto body = body_status.ConsumeValueOrDie();

  auto gather_while = builder->While(cond, body, init);
  return builder->GetTupleElement(gather_while, 3);
}

namespace {

class UnsortedSegmentSum : public XlaOpKernel {
 public:
  explicit UnsortedSegmentSum(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // output = unsorted_segment_sum(data, indices, num_segments)
    // Compute a tensor such that:
    //    output[i] = sum over {j where indices[j] == i} of data[j]
    //    output[i] == 0 if i does not appear in indices
    //
    // Contrast with segment_sum(), which assumes indices are sorted and that
    // max(indices)+1 is the desired size of the output.
    //
    // The returned output tensor has the same type as data, and the same shape
    // as data with the first indices.rank dimensions are replaced
    // by a single dimension with size num_segments.
    auto data = ctx->Input(0);
    auto data_shape = ctx->InputShape(0);

    auto indices = ctx->Input(1);
    auto indices_shape = ctx->InputShape(1);

    int64 num_segments;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(2, &num_segments));

    OP_REQUIRES(ctx, data_shape.dims() >= indices_shape.dims(),
                errors::InvalidArgument(
                    "UnsortedSegmentSum requires that indices' rank be"
                    " less than or equal to data's rank."));
    // Validate that indices.shape is a prefix of data.shape.
    for (int d = 0; d < indices_shape.dims(); ++d) {
      OP_REQUIRES(ctx, (data_shape.dim_size(d) == indices_shape.dim_size(d)),
                  errors::InvalidArgument(
                      "UnsortedSegmentSum requires indices shape to be prefix"
                      " of data_shape, but dimension ",
                      d, " differs ", data_shape.dim_size(d), " vs. ",
                      indices_shape.dim_size(d)));
    }
    auto result = XlaComputeScatterAddDynamicSlice(
        ctx, data, data_shape, indices, indices_shape, num_segments, dtype_,
        ctx->builder());
    ctx->SetOutput(0, result);
  }

 private:
  DataType dtype_;
};

REGISTER_XLA_OP(Name("UnsortedSegmentSum"), UnsortedSegmentSum);

}  // namespace
}  // namespace tensorflow
