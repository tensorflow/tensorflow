/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_OUTFEED_OPS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_OUTFEED_OPS_H_

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tpu/kernels/transfer_ops.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tensorflow {

// The OutfeedDequeue op is used to retrieve a single tensor from the device
// outfeed queue.
template <class T>
class TpuOutfeedDequeueOp : public T {
 public:
  explicit TpuOutfeedDequeueOp(
      OpKernelConstruction* ctx,
      std::unique_ptr<TpuTransferOpInterface> transfer_op)
      : T(ctx, "outfeed_dequeue", 1, std::move(transfer_op)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape_, &xla_shape_));
  }

  Status DoWork(OpKernelContext* ctx, int device_ordinal) override {
    Tensor* output;
    TF_RETURN_IF_ERROR(ctx->allocate_output(0, shape_, &output));

    // Transfer from the outfeed interface of the device.
    xla::MutableBorrowingLiteral literal;
    TF_RETURN_IF_ERROR(
        HostTensorToMutableBorrowingLiteral(xla_shape_, output, &literal));

    VLOG(1) << "TransferLiteralFromOutfeed "
            << xla::ShapeUtil::HumanStringWithLayout(xla_shape_);

    TF_RETURN_IF_ERROR(
        T::transfer_op_->TransferLiteralFromOutfeed(device_ordinal, literal));

    VLOG(1) << "TransferLiteralFromOutfeed complete.";

    return Status::OK();
  }

 private:
  TensorShape shape_;
  DataType dtype_;
  xla::Shape xla_shape_;

  // OutfeedDequeueOp is neither copyable nor movable.
  TpuOutfeedDequeueOp(const TpuOutfeedDequeueOp&) = delete;
  TpuOutfeedDequeueOp& operator=(const TpuOutfeedDequeueOp&) = delete;
};

// The OutfeedDequeueTuple op is used to retrieve multiple tensors from the
// device outfeed queue.
template <class T>
class TpuOutfeedDequeueTupleOp : public T {
 public:
  explicit TpuOutfeedDequeueTupleOp(
      OpKernelConstruction* ctx,
      std::unique_ptr<TpuTransferOpInterface> transfer_op)
      : T(ctx, "outfeed_dequeue", 1, std::move(transfer_op)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
    OP_REQUIRES(
        ctx, shapes_.size() == dtypes_.size(),
        errors::InvalidArgument("shapes and dtypes must be the same length."));
    // The `dtypes` list is inferred from the supplied inputs, so it
    // is always the correct length.
    for (int i = 0; i < shapes_.size(); i++) {
      xla::Shape xla_shape;
      OP_REQUIRES_OK(ctx,
                     TensorShapeToXLAShape(dtypes_[i], shapes_[i], &xla_shape));
      xla_shapes_.push_back(xla_shape);
    }
    tuple_shape_ = xla::ShapeUtil::MakeTupleShape(xla_shapes_);
  }

  Status DoWork(OpKernelContext* ctx, int device_ordinal) override {
    VLOG(1) << "TransferLiteralFromOutfeed "
            << xla::ShapeUtil::HumanStringWithLayout(tuple_shape_);

    for (int i = 0; i < shapes_.size(); ++i) {
      Tensor* output;
      TF_RETURN_IF_ERROR(ctx->allocate_output(i, shapes_[i], &output));

      xla::MutableBorrowingLiteral literal;
      TF_RETURN_IF_ERROR(HostTensorToMutableBorrowingLiteral(xla_shapes_[i],
                                                             output, &literal));
      TF_RETURN_IF_ERROR(
          T::transfer_op_->TransferLiteralFromOutfeed(device_ordinal, literal));
    }
    return Status::OK();
  }

 private:
  std::vector<TensorShape> shapes_;
  DataTypeVector dtypes_;
  std::vector<xla::Shape> xla_shapes_;
  xla::Shape tuple_shape_;

  // OutfeedDequeueTupleOp is neither copyable nor movable.
  TpuOutfeedDequeueTupleOp(const TpuOutfeedDequeueTupleOp&) = delete;
  TpuOutfeedDequeueTupleOp& operator=(const TpuOutfeedDequeueTupleOp&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_OUTFEED_OPS_H_
