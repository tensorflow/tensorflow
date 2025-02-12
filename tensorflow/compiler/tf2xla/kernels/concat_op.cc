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

// XLA-specific Concat Ops.

#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/kernels/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// --------------------------------------------------------------------------
class ConcatBaseOp : public XlaOpKernel {
 public:
  ConcatBaseOp(OpKernelConstruction* c, int64_t axis_index)
      : XlaOpKernel(c), axis_index_(axis_index) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape concat_dim_tensor_shape = ctx->InputShape(axis_index_);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(concat_dim_tensor_shape),
                errors::InvalidArgument(
                    "Concat dim tensor should be a scalar, but got shape ",
                    concat_dim_tensor_shape.DebugString()));
    int64_t concat_dim;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntScalar(axis_index_, &concat_dim));

    std::vector<xla::XlaOp> values;
    std::vector<TensorShape> shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("values", &values, &shapes));
    const int N = values.size();
    const int input_dims = shapes[0].dims();
    const TensorShape& input_shape = shapes[0];

    int64_t axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(ctx, 0 <= axis && axis < input_dims,
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));

    // Make a vector holding the XlaOp for each of the inputs that has non-zero
    // elements.
    std::vector<xla::XlaOp> input_data;
    int output_concat_dim = 0;
    for (int i = 0; i < N; ++i) {
      xla::XlaOp handle = values[i];
      const TensorShape& in_shape = shapes[i];
      OP_REQUIRES(
          ctx, in_shape.dims() == input_dims,
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", in_shape.DebugString()));
      if (in_shape.dims() == 0) {
        // Inputs that come in as scalars must be reshaped to 1-vectors.
        input_data.push_back(xla::Reshape(handle, {1}));
      } else {
        input_data.push_back(handle);
      }
      output_concat_dim += in_shape.dims() > 0 ? in_shape.dim_size(axis) : 1;
    }

    VLOG(1) << "Concat dim " << concat_dim << " equivalent to " << axis;
    ctx->SetOutput(0, xla::ConcatInDim(ctx->builder(), input_data, axis));
  }

 private:
  int axis_index_;
};

class ConcatOp : public ConcatBaseOp {
 public:
  explicit ConcatOp(OpKernelConstruction* c)
      : ConcatBaseOp(c, /* axis_index */ 0) {}
};

// ConcatV2 operation is the same as Concat except 'concat_dim'
// is the last input instead of the first and renamed to 'axis'.
class ConcatV2Op : public ConcatBaseOp {
 public:
  explicit ConcatV2Op(OpKernelConstruction* c)
      : ConcatBaseOp(c, /* axis_index */ c->num_inputs() - 1) {}
};

REGISTER_XLA_OP(Name("Concat").CompileTimeConstantInput("concat_dim"),
                ConcatOp);
REGISTER_XLA_OP(Name("ConcatV2")
                    .TypeConstraint("Tidx", {DT_INT32, DT_INT64})
                    .CompileTimeConstantInput("axis"),
                ConcatV2Op);

class ConcatOffsetOp : public XlaOpKernel {
 public:
  explicit ConcatOffsetOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape_type", &shape_type_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape concat_dim_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(concat_dim_shape),
                errors::InvalidArgument(
                    "Concat dim tensor should be a scalar, but got shape ",
                    concat_dim_shape.DebugString()));
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(ctx->InputShape(i)),
                  errors::InvalidArgument("input ", i,
                                          " should be a vector, but got shape ",
                                          ctx->InputShape(i).DebugString()));
    }
    // Suppose a Concat() op needs to Concatenate N tensors, each of
    // which has the same number of dimensions.  Their shapes match
    // except the concat dimension.
    //
    // E.g., say, we want to concatenate 3 tensors in the 2nd
    // dimension, and their shapes are:
    //
    //  [2, 2, 5, 7]
    //  [2, 3, 5, 7]
    //  [2, 4, 5, 7]
    //
    // Here, N=3, cdim=1, dims=4. The concatenated tensor has shape
    // [2,9,5,7]. We will compute the cumulative sum along the 2nd
    // dimension to figure out each input's offset in the concatenated
    // output:
    //  [0, 0, 0, 0]
    //  [0, 2, 0, 0]
    //  [0, 5, 0, 0]
    const int32_t N = ctx->num_inputs() - 1;
    const TensorShape inp0_shape = ctx->InputShape(1);
    std::vector<int64_t> inp0_dims;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntVector(
                       1, &inp0_dims, xla::ValueInferenceMode::kUpperBound));
    const int64_t inp0_rank = inp0_shape.num_elements();

    int64_t cdim;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(0, &cdim));

    VLOG(1) << "ConcatOffset " << cdim << "," << inp0_rank;
    int32_t axis = cdim < 0 ? cdim + inp0_rank : cdim;
    OP_REQUIRES(ctx, FastBoundsCheck(axis, inp0_rank),
                errors::InvalidArgument("Concat dim is out of range: ", axis,
                                        " vs. ", inp0_rank));
    int64_t offset = 0;
    for (int i = 0; i < N; ++i) {
      const TensorShape inp_shape = ctx->InputShape(1 + i);
      OP_REQUIRES(ctx, inp0_rank == inp_shape.num_elements(),
                  errors::InvalidArgument("input ", i, " should contain ",
                                          inp0_rank, " elements, but got ",
                                          inp_shape.num_elements()));
      std::vector<int64_t> inp_dims;
      OP_REQUIRES_OK(
          ctx, ctx->ConstantInputAsIntVector(
                   1 + i, &inp_dims, xla::ValueInferenceMode::kUpperBound));

      std::vector<int64_t> output_dims(inp0_rank);
      for (int64_t j = 0; j < inp0_rank; ++j) {
        if (j == axis) {
          output_dims[j] = offset;
          offset += inp_dims[j];
        } else {
          const int64_t inp0_element = inp0_dims[j];
          const int64_t inp_element = inp_dims[j];
          OP_REQUIRES(ctx, inp0_element == inp_element,
                      errors::InvalidArgument(
                          "All dimensions except ", axis, " must match. Input ",
                          i, " has shape [", absl::StrJoin(inp_dims, " "),
                          "] and doesn't match input 0 with shape [",
                          absl::StrJoin(inp0_dims, " "), "]."));
          output_dims[j] = 0;
        }
      }
      TensorShape out_shape;
      OP_REQUIRES_OK(ctx,
                     TensorShape::BuildTensorShape(output_dims, &out_shape));
      Tensor out_constant(shape_type_, TensorShape({inp0_rank}));
      OP_REQUIRES_OK(ctx, TensorShapeToConstant(out_shape, &out_constant));

      ctx->SetConstantOutput(i, out_constant);
    }
  }

 private:
  DataType shape_type_;
};

REGISTER_XLA_OP(Name("ConcatOffset")
                    .TypeConstraint("shape_type", {DT_INT32, DT_INT64})
                    .CompileTimeConstantInput("concat_dim")
                    .CompileTimeConstantInput("shape"),
                ConcatOffsetOp);

}  // namespace
}  // namespace tensorflow
