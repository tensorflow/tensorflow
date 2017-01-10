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

// XLA-specific dynamic stitch Op.

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {
namespace {

class DynamicStitchOp : public XlaOpKernel {
 public:
  explicit DynamicStitchOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES(
        ctx, ctx->num_inputs() > 0,
        errors::InvalidArgument("DynamicStitchOp: Must have some inputs"));
    OP_REQUIRES(ctx, ctx->num_inputs() % 2 == 0,
                errors::InvalidArgument(
                    "DynamicStitchOp: Must have even number of arguments"));
    // Compute expected input signature
    const int n = ctx->num_inputs() / 2;
    const DataType dt = ctx->input_type(n);
    DataTypeVector expected;
    for (int i = 0; i < n; i++) {
      expected.push_back(DT_INT32);
    }
    for (int i = 0; i < n; i++) {
      expected.push_back(dt);
    }
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected, {dt}));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Validate that data_shape[i] = indices[i].shape() + constant
    std::vector<xla::Literal> indices_input;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputList("indices", &indices_input));

    std::vector<xla::ComputationDataHandle> data;
    std::vector<TensorShape> data_shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("data", &data, &data_shapes));

    std::vector<xla::Literal> indices(indices_input.size());

    const TensorShape& data0_shape = data_shapes[0];
    const TensorShape indices0_shape =
        XLAShapeToTensorShape(indices_input[0].shape());
    for (int input_num = 0; input_num < indices_input.size(); input_num++) {
      const TensorShape indices_shape =
          XLAShapeToTensorShape(indices_input[input_num].shape());
      const TensorShape& data_shape = data_shapes[input_num];
      OP_REQUIRES(ctx, TensorShapeUtils::StartsWith(data_shape, indices_shape),
                  errors::InvalidArgument(
                      "data[", input_num, "].shape = ",
                      data_shape.DebugString(), " does not start with indices[",
                      input_num, "].shape = ", indices_shape.DebugString()));
      OP_REQUIRES(ctx,
                  input_num == 0 || SameExtraShape(data0_shape, indices0_shape,
                                                   data_shape, indices_shape),
                  errors::InvalidArgument(
                      "Need data[0].shape[", indices0_shape.dims(),
                      ":] = data[", input_num, "].shape[", indices_shape.dims(),
                      ":], got data[0].shape = ", data0_shape.DebugString(),
                      ", data[", input_num, "].shape = ",
                      data_shape.DebugString(), ", indices[0].shape = ",
                      indices0_shape.DebugString(), ", indices[", input_num,
                      "].shape = ", indices_shape.DebugString()));

      OP_REQUIRES_OK(ctx,
                     XlaHelpers::ReshapeLiteral(indices_input[input_num],
                                                {indices_shape.num_elements()},
                                                &indices[input_num]));
    }

    // Find which slice will be used for each index. If the same index
    // appears in multiple inputs, the last one is used. The logic
    // here is different from that in third_party/tensorflow because
    // it is important for XLA that there be a well-formed Concat
    // operation at the end. The existing CPU/GPU code copies multiple
    // source slices to the same destination slice if there are
    // repeated indices, whereas the XLA code works out which
    // source slice will 'win' and only uses that in the Concat.
    int max_index = -1;
    for (int input_num = 0; input_num < indices.size(); input_num++) {
      for (int i = 0; i < indices[input_num].shape().dimensions(0); ++i) {
        max_index = std::max(
            max_index, xla::LiteralUtil::Get<int>(indices[input_num], {i}));
      }
    }
    int number_of_indices = max_index + 1;
    OP_REQUIRES(ctx, number_of_indices > 0,
                errors::InvalidArgument("no indices supplied"));
    // Construct the reverse mapping, for each index, of which slice of which
    // input it comes from.
    std::vector<int32> src_input_vector(number_of_indices);
    std::vector<int32> src_slice_vector(number_of_indices);
    std::vector<bool> src_index_used(number_of_indices);
    int index_used_count = 0;
    for (int input_num = 0; input_num < indices.size(); input_num++) {
      for (int i = 0; i < indices[input_num].shape().dimensions(0); ++i) {
        int index = xla::LiteralUtil::Get<int>(indices[input_num], {i});
        src_input_vector[index] = input_num;
        src_slice_vector[index] = i;
        if (!src_index_used[index]) {
          src_index_used[index] = true;
          ++index_used_count;
        }
      }
    }
    OP_REQUIRES(ctx, index_used_count == number_of_indices,
                errors::InvalidArgument("not all indices are used"));

    // Look up all the children expressions that represent the data
    // inputs.
    std::vector<xla::ComputationDataHandle> input(indices.size());
    for (int input_num = 0; input_num < indices.size(); input_num++) {
      TensorShape new_shape;
      // first reshaped dimension is the number of indices for this input.
      new_shape.AddDim(indices[input_num].shape().dimensions(0));
      // Then the rest are the common extra shape.
      for (int d = indices0_shape.dims(); d < data0_shape.dims(); d++) {
        new_shape.AddDim(data0_shape.dim_size(d));
      }
      // Get the data, shaped appropriately.
      auto handle = data[input_num];
      if (new_shape == data_shapes[input_num]) {
        input[input_num] = handle;
      } else {
        input[input_num] =
            ctx->builder()->Reshape(handle, new_shape.dim_sizes());
      }
    }

    // Set up the vectors for slicing: the first dimension will vary
    // slice by slice, and the rest take the full common extra shape.
    std::vector<int64> slice_start(1 + data0_shape.dims() -
                                   indices0_shape.dims());
    std::vector<int64> slice_limit(1 + data0_shape.dims() -
                                   indices0_shape.dims());
    for (int d = indices0_shape.dims(); d < data0_shape.dims(); d++) {
      slice_limit[1 + d - indices0_shape.dims()] = data0_shape.dim_size(d);
    }
    std::vector<xla::ComputationDataHandle> to_concat(number_of_indices);
    for (int index_num = 0; index_num < number_of_indices; index_num++) {
      const auto& expression = input[src_input_vector[index_num]];
      // Take the appropriate slice of data.
      slice_start[0] = src_slice_vector[index_num];
      slice_limit[0] = src_slice_vector[index_num] + 1;
      // And place it in the concat list in the place indicated by
      // the index.
      to_concat[index_num] =
          ctx->builder()->Slice(expression, slice_start, slice_limit);
    }

    ctx->SetOutput(0, ctx->builder()->ConcatInDim(to_concat, 0));
  }

 private:
  // Check if data0_shape[indices0.dims():] == data1_shape[indices1.dims():]
  static bool SameExtraShape(const TensorShape& data0_shape,
                             const TensorShape& indices0,
                             const TensorShape& data1_shape,
                             const TensorShape& indices1) {
    const int extra0 = data0_shape.dims() - indices0.dims();
    const int extra1 = data1_shape.dims() - indices1.dims();
    if (extra0 != extra1) return false;
    for (int i = 0; i < extra0; i++) {
      if (data0_shape.dim_size(indices0.dims() + i) !=
          data1_shape.dim_size(indices1.dims() + i)) {
        return false;
      }
    }
    return true;
  }
};

REGISTER_XLA_OP("DynamicStitch", DynamicStitchOp);

}  // namespace
}  // namespace tensorflow
