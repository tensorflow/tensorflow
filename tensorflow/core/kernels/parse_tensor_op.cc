/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/parsing_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

class ParseTensorOp : public OpKernel {
 public:
  explicit ParseTensorOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& serialized = ctx->input(0);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(serialized.shape()),
                errors::InvalidArgument(
                    "Expected `serialized` to be a scalar, got shape: ",
                    serialized.shape().DebugString()));

    auto serialized_t = serialized.scalar<string>();

    TensorProto proto;
    OP_REQUIRES(ctx, ParseProtoUnlimited(&proto, serialized_t()),
                errors::InvalidArgument(
                    "Could not parse `serialized` as TensorProto: '",
                    serialized_t(), "'"));

    Tensor output;
    OP_REQUIRES_OK(ctx, ctx->device()->MakeTensorFromProto(
                            proto, ctx->output_alloc_attr(0), &output));

    OP_REQUIRES(
        ctx, out_type_ == output.dtype(),
        errors::InvalidArgument("Type mismatch between parsed tensor (",
                                DataTypeString(output.dtype()), ") and dtype (",
                                DataTypeString(out_type_), ")"));

    ctx->set_output(0, output);
  }

 private:
  DataType out_type_;
};

REGISTER_KERNEL_BUILDER(Name("ParseTensor").Device(DEVICE_CPU), ParseTensorOp);

template <typename T>
class SerializeTensorOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor = context->input(0);
    TensorProto proto;
    if (tensor.dtype() == DT_STRING) {
      tensor.AsProtoField(&proto);
    } else {
      tensor.AsProtoTensorContent(&proto);
    }
    Tensor* proto_string = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &proto_string));
    CHECK(proto.SerializeToString(&proto_string->scalar<string>()()));
  }
};

#define REGISTER(T)                                                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SerializeTensor").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SerializeTensorOp<T>);
TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER

}  // namespace tensorflow
