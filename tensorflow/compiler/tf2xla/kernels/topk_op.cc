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
    const xla::XlaOp input = context->Input(0);
    xla::XlaOp iota;
    OP_REQUIRES_OK(context, XlaHelpers::Iota(b, DT_INT32, n, &iota));

    // TODO(b/73891930): add a key-value sort to HLO, rather than using
    // bit-packing tricks here.
    // TODO(b/73891930): this implementation will convert Infs to NaNs. A
    // key-value sort would avoid this; for now, it is no worse than, say, the
    // CPU backend in fast-math mode.

    // Pack elements as:
    // * upper 16 bits are the value
    // * lower 16 bits are the index.
    xla::XlaOp packed = b->BitcastConvertType(
        b->Or(b->BitcastConvertType(b->ConvertElementType(input, xla::F32),
                                    xla::S32),
              iota),
        xla::F32);

    // TODO(phawkins): use a more efficient algorithm that does not require a
    // full sort.
    xla::XlaOp sorted = b->Slice(b->Rev(b->Sort(packed), {0}),
                                 /*start_indices=*/{0},
                                 /*limit_indices=*/{k},
                                 /*strides=*/{1});

    // Unpack the value/index
    xla::XlaOp x = b->BitcastConvertType(sorted, xla::S32);
    xla::XlaOp indices = b->And(x, b->ConstantR0<int32>(0x0000FFFF));
    xla::XlaOp values = b->ConvertElementType(
        b->BitcastConvertType(b->And(x, b->ConstantR0<int32>(0xFFFF0000)),
                              xla::F32),
        xla::BF16);

    context->SetOutput(0, values);
    context->SetOutput(1, indices);
  }

 private:
  bool sorted_;
};

REGISTER_XLA_OP(
    Name("TopKV2").CompileTimeConstInput("k").TypeConstraint("T", DT_BFLOAT16),
    TopKOp);

}  // namespace
}  // namespace tensorflow
