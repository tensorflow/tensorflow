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

#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/stream_executor_util.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"

#include "absl/container/flat_hash_set.h"

namespace tensorflow {

class PopDatastreamInfeedEnqueueOp : public XlaOpKernel {
 public:
  explicit PopDatastreamInfeedEnqueueOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->num_inputs() == 1,
                errors::InvalidArgument(absl::StrFormat(
                    "Enqueue supports only single tensor input")));

    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    const TensorShape input_shape = ctx->InputShape(0);

    xla::Shape xla_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(0));

    xla::XlaBuilder* b = ctx->builder();

    xla::XlaOp infeed_op = xla::Infeed(b, xla_shape, "enqueue");
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopDatastreamInfeedEnqueueOp);
};

REGISTER_IPU_OP("PopDatastreamInfeedEnqueue", PopDatastreamInfeedEnqueueOp);

class PopDatastreamInfeedDequeueOp : public XlaOpKernel {
 public:
  explicit PopDatastreamInfeedDequeueOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &type_));
  }

  ~PopDatastreamInfeedDequeueOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::Shape xla_shape;
    auto status = TensorShapeToXLAShape(type_, shape_, &xla_shape);
    if (status != Status::OK()) {
      VLOG(1) << "shape conversion error";
    }

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp infeed_op = xla::Infeed(b, xla_shape, "dequeue");
    ctx->SetOutput(0, infeed_op);
  }

 private:
  TensorShape shape_;
  tensorflow::DataType type_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopDatastreamInfeedDequeueOp);
};

REGISTER_IPU_OP("PopDatastreamInfeedDequeue", PopDatastreamInfeedDequeueOp);

}  // namespace tensorflow
