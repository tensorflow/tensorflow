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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/no_op.h"

namespace tensorflow {
namespace {

class TopKOp : public XlaOpKernel {
 public:
  explicit TopKOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sorted", &sorted_));
  }

  void Compile(XlaOpKernelContext* context) override {
    int64 k;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(1, &k));
    OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));
    const TensorShape input_shape = context->InputShape(0);
    OP_REQUIRES(context, input_shape.dims() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input_shape.DebugString()));
    OP_REQUIRES(
        context, input_shape.dim_size(input_shape.dims() - 1) >= k,
        errors::InvalidArgument("input must have at least k columns. Had ",
                                input_shape.dim_size(input_shape.dims() - 1),
                                ", needed ", k));

    OP_REQUIRES(
        context, input_shape.dims() == 1,
        errors::Unimplemented("TopK is implemented for 1-D inputs, got shape ",
                              input_shape.DebugString()));

    const int64 n = input_shape.dim_size(0);
    OP_REQUIRES(context, n < (1 << 16),
                errors::Unimplemented(
                    "TopK is implemented for sizes up to 2**16, got shape ",
                    input_shape.DebugString()));

    xla::XlaBuilder* const b = context->builder();
    if (input_shape.dim_size(0) < k) {
      k = input_shape.dim_size(0);
    }
    const xla::XlaOp input_bf16 = context->Input(0);
    xla::XlaOp iota_s32;
    OP_REQUIRES_OK(context, XlaHelpers::Iota(b, DT_INT32, n, &iota_s32));

    // TODO(b/73891930): add a key-value sort to HLO, rather than using
    // bit-packing tricks here.

    xla::XlaOp zero = xla::ConstantR0<int32>(b, 0);

    // max can either be 0x7FFFFFFF or 0x8000000. Neither choice is totally
    // ideal. The implications of the choice are:
    //
    // 0x7FFFFFFF
    // 1. +0.0 > -0.0
    // 2. The elements of the inputs and outputs are bitwise identical.
    // 3. The sort is unstable since a later +0.0 will appear before an earlier
    // -0.0.
    //
    // 0x8000000
    // 1. +0.0 == -0.0
    // 2. All -0.0 in the input are replaced with +0.0 in the output.
    // 3. The sort is stable.
    xla::XlaOp max = xla::ConstantR0<int32>(b, 0x80000000);
    xla::XlaOp index_mask = xla::ConstantR0<int32>(b, 0x0000FFFF);
    xla::XlaOp value_mask = xla::ConstantR0<int32>(b, 0xFFFF0000);

    // Convert to from bf16 to f32. The lower 16-bits are zero due to the
    // definition of bf16.
    xla::XlaOp input_f32 = xla::ConvertElementType(input_bf16, xla::F32);

    // Negate the input to reverse sort it. The lower 16-bits are zero, because
    // negating a float is just inverting the high-bit.
    xla::XlaOp negative_input_f32 = xla::Neg(input_f32);

    // Convert to a sign magnitude integer. The lower 16-bits are zero, since
    // bitcast convert doesn't change any bits.
    xla::XlaOp negative_input_sm32 =
        xla::BitcastConvertType(negative_input_f32, xla::S32);

    // Convert from sign magnitude integer to two's complement integer. The
    // lower 16-bits are zero on both sides of the select. On the false side,
    // the value is unchanged, and on the true side, the lower 16-bits of max
    // are all zero, so the lower 16-bits of the result of the subtraction will
    // also be zero.
    xla::XlaOp negative_input_s32 =
        xla::Select(xla::Lt(negative_input_sm32, zero),
                    xla::Sub(max, negative_input_sm32), negative_input_sm32);

    // In order for the Or with iota_s32 to to work properly, the lower 16-bits
    // of negative_input_32 must be zero.

    // Pack elements as:
    // * upper 16 bits are the value
    // * lower 16 bits are the index.
    xla::XlaOp packed_s32 = xla::Or(negative_input_s32, iota_s32);

    // TODO(phawkins): use a more efficient algorithm that does not require a
    // full sort.
    xla::XlaOp sorted_s32 = xla::Slice(xla::Sort(packed_s32),
                                       /*start_indices=*/{0},
                                       /*limit_indices=*/{k},
                                       /*strides=*/{1});

    // Unpack the value/index.
    xla::XlaOp indices_s32 = xla::And(sorted_s32, index_mask);
    xla::XlaOp negative_values_s32 = xla::And(sorted_s32, value_mask);

    // Convert from two's complement integer to sign magnitude integer.
    xla::XlaOp negative_values_sm32 =
        xla::Select(xla::Lt(negative_values_s32, zero),
                    xla::Sub(max, negative_values_s32), negative_values_s32);

    xla::XlaOp negative_values_f32 =
        xla::BitcastConvertType(negative_values_sm32, xla::F32);

    // Negate the values to get back the original inputs.
    xla::XlaOp values_f32 = xla::Neg(negative_values_f32);

    // Convert from f32 to bf16.
    xla::XlaOp values_bf16 = xla::ConvertElementType(values_f32, xla::BF16);

    context->SetOutput(0, values_bf16);
    context->SetOutput(1, indices_s32);
  }

 private:
  bool sorted_;
};

REGISTER_XLA_OP(
    Name("TopKV2").CompileTimeConstInput("k").TypeConstraint("T", DT_BFLOAT16),
    TopKOp);

}  // namespace
}  // namespace tensorflow
