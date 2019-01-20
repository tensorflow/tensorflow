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

#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
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
#include "tensorflow/compiler/xla/service/transfer_manager.h"

#include "absl/container/flat_hash_set.h"

namespace tensorflow {

class PopDatastreamInfeedEnqueueOp : public OpKernel {
 public:
  explicit PopDatastreamInfeedEnqueueOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), device_ordinal_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));

    OP_REQUIRES(ctx, device_ordinal_ >= 0,
                errors::InvalidArgument("Need device_ordinal >= 0, got ",
                                        device_ordinal_));
  }
  ~PopDatastreamInfeedEnqueueOp() override{};

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());
    auto* p =
        static_cast<xla::poplarplugin::PoplarPlatform*>(platform.ValueOrDie());

    auto executor = p->ExecutorForDevice(device_ordinal_).ValueOrDie();

    const Tensor& input = ctx->input(0);
    const TensorShape& tensor_shape = input.shape();
    const DataType& tensor_dtype = input.dtype();
    tensorflow::StringPiece tensor_data = input.tensor_data();

    xla::PrimitiveType xla_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(tensor_dtype, &xla_type));
    auto xla_shape = TensorShapeToXLAShape(xla_type, tensor_shape);

    auto* transfer_manager =
        xla::TransferManager::GetForPlatform(p).ValueOrDie();
    auto literal_input = xla::BorrowingLiteral(tensor_data.data(), xla_shape);
    transfer_manager->TransferLiteralToInfeed(executor, literal_input);
  }

 private:
  int device_ordinal_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopDatastreamInfeedEnqueueOp);
};

REGISTER_KERNEL_BUILDER(Name("PopDatastreamInfeedEnqueue").Device(DEVICE_CPU),
                        PopDatastreamInfeedEnqueueOp);

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
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(type_, shape_, &xla_shape));
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
