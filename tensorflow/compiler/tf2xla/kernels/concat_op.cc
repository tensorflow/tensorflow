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

#include <limits>
#include <vector>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// --------------------------------------------------------------------------
class ConcatBaseOp : public XlaOpKernel {
 public:
  ConcatBaseOp(OpKernelConstruction* c, int axis_index)
      : XlaOpKernel(c), axis_index_(axis_index) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape concat_dim_tensor_shape = ctx->InputShape(axis_index_);
    OP_REQUIRES(
        ctx, IsLegacyScalar(concat_dim_tensor_shape),
        errors::InvalidArgument(
            "Concat dim tensor should be a scalar integer, but got shape ",
            concat_dim_tensor_shape.DebugString()));
    xla::Literal literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(axis_index_, &literal));
    // TODO(annarev): add a helper to support int64 input.
    const int32 concat_dim = xla::LiteralUtil::Get<int>(literal, {});

    std::vector<xla::ComputationDataHandle> values;
    std::vector<TensorShape> shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("values", &values, &shapes));
    const int N = values.size();
    const int input_dims = shapes[0].dims();
    const TensorShape& input_shape = shapes[0];

    int32 axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(ctx,
                (0 <= axis && axis < input_dims) ||
                    (allow_legacy_scalars() && concat_dim == 0),
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));

    // Make a vector holding the ComputationDataHandles for each of
    // the inputs that has non-zero elements.
    std::vector<xla::ComputationDataHandle> input_data;
    int output_concat_dim = 0;
    const bool input_is_scalar = IsLegacyScalar(input_shape);
    for (int i = 0; i < N; ++i) {
      xla::ComputationDataHandle handle = values[i];
      const TensorShape& in_shape = shapes[i];
      const bool in_is_scalar = IsLegacyScalar(in_shape);
      OP_REQUIRES(
          ctx,
          in_shape.dims() == input_dims || (input_is_scalar && in_is_scalar),
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i, "] = ",
              in_shape.DebugString()));
      if (in_shape.dims() == 0) {
        // Inputs that come in as scalars must be reshaped to 1-vectors.
        input_data.push_back(ctx->builder()->Reshape(handle, {1}));
      } else {
        input_data.push_back(handle);
      }
      output_concat_dim += in_shape.dims() > 0 ? in_shape.dim_size(axis) : 1;
    }

    VLOG(1) << "Concat dim " << concat_dim << " equivalent to " << axis;
    ctx->SetOutput(0, ctx->builder()->ConcatInDim(input_data, axis));
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

REGISTER_XLA_OP("Concat", ConcatOp);
REGISTER_XLA_OP("ConcatV2", ConcatV2Op);

class ConcatOffsetOp : public XlaOpKernel {
 public:
  explicit ConcatOffsetOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape concat_dim_shape = ctx->InputShape(0);
    OP_REQUIRES(
        ctx, IsLegacyScalar(concat_dim_shape),
        errors::InvalidArgument(
            "Concat dim tensor should be a scalar integer, but got shape ",
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
    const int32 N = ctx->num_inputs() - 1;
    const TensorShape inp0_shape = ctx->InputShape(1);
    xla::Literal inp0_literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(1, &inp0_literal));
    const int64 dims = inp0_shape.num_elements();

    xla::Literal concat_dim_literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(0, &concat_dim_literal));
    const int64 cdim = xla::LiteralUtil::Get<int>(concat_dim_literal, {});

    VLOG(1) << "ConcatOffset " << cdim << "," << dims;
    int32 axis = cdim < 0 ? cdim + dims : cdim;
    OP_REQUIRES(ctx, FastBoundsCheck(axis, dims),
                errors::InvalidArgument("Concat dim is out of range: ", axis,
                                        " vs. ", dims));
    int32 offset = 0;
    for (int i = 0; i < N; ++i) {
      const TensorShape inp_shape = ctx->InputShape(1 + i);
      OP_REQUIRES(ctx, dims == inp_shape.num_elements(),
                  errors::InvalidArgument("input ", i, " should contain ", dims,
                                          " elements, but got ",
                                          inp_shape.num_elements()));
      xla::Literal inp_literal;
      OP_REQUIRES_OK(ctx, ctx->ConstantInput(1 + i, &inp_literal));

      Tensor out_constant(DT_INT32, TensorShape({dims}));
      auto out_vec = out_constant.vec<int32>();
      for (int64 j = 0; j < dims; ++j) {
        if (j == axis) {
          out_vec(j) = offset;
          offset += xla::LiteralUtil::Get<int>(inp_literal, {j});
        } else {
          const int32 inp0_element =
              xla::LiteralUtil::Get<int>(inp0_literal, {j});
          const int32 inp_element =
              xla::LiteralUtil::Get<int>(inp_literal, {j});
          OP_REQUIRES(
              ctx, (inp0_element == inp_element),
              errors::InvalidArgument("input[", i, ",", j, "] mismatch: ",
                                      inp0_element, " vs. ", inp_element));
          out_vec(j) = 0;
        }
      }

      ctx->SetConstantOutput(i, out_constant);
    }
  }
};

REGISTER_XLA_OP("ConcatOffset", ConcatOffsetOp);

}  // namespace
}  // namespace tensorflow
