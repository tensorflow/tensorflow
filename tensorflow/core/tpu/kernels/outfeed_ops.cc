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

#include "tensorflow/core/tpu/kernels/outfeed_ops.h"

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/tpu/kernels/transfer_ops.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tensorflow {
namespace {
template <class T>
class StreamExecutorOutfeedDequeueOp : public TpuOutfeedDequeueOp<T> {
 public:
  explicit StreamExecutorOutfeedDequeueOp(OpKernelConstruction* ctx)
      : TpuOutfeedDequeueOp<T>(
            ctx, absl::make_unique<StreamExecutorTransferOpImpl>()) {}

 private:
  StreamExecutorOutfeedDequeueOp(const StreamExecutorOutfeedDequeueOp&) =
      delete;
  StreamExecutorOutfeedDequeueOp& operator=(
      const StreamExecutorOutfeedDequeueOp&) = delete;
};

template <class T>
class StreamExecutorOutfeedDequeueTupleOp : public TpuOutfeedDequeueTupleOp<T> {
 public:
  explicit StreamExecutorOutfeedDequeueTupleOp(OpKernelConstruction* ctx)
      : TpuOutfeedDequeueTupleOp<T>(
            ctx, absl::make_unique<StreamExecutorTransferOpImpl>()) {}

 private:
  StreamExecutorOutfeedDequeueTupleOp(
      const StreamExecutorOutfeedDequeueTupleOp&) = delete;
  StreamExecutorOutfeedDequeueTupleOp& operator=(
      const StreamExecutorOutfeedDequeueTupleOp&) = delete;
};

}  // namespace

template <class T>
TpuOutfeedDequeueOp<T>::TpuOutfeedDequeueOp(
    OpKernelConstruction* ctx,
    std::unique_ptr<TpuTransferOpInterface> transfer_op)
    : T(ctx, "outfeed_dequeue", 1, std::move(transfer_op)) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape_, &xla_shape_));
}

template <class T>
Status TpuOutfeedDequeueOp<T>::DoWork(OpKernelContext* ctx,
                                      int device_ordinal) {
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

// The OutfeedDequeueTuple op is used to retrieve multiple tensors from the
// device outfeed queue.
template <class T>
TpuOutfeedDequeueTupleOp<T>::TpuOutfeedDequeueTupleOp(
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

template <class T>
Status TpuOutfeedDequeueTupleOp<T>::DoWork(OpKernelContext* ctx,
                                           int device_ordinal) {
  VLOG(1) << "TransferLiteralFromOutfeed "
          << xla::ShapeUtil::HumanStringWithLayout(tuple_shape_);

  for (int i = 0; i < shapes_.size(); ++i) {
    Tensor* output;
    TF_RETURN_IF_ERROR(ctx->allocate_output(i, shapes_[i], &output));

    xla::MutableBorrowingLiteral literal;
    TF_RETURN_IF_ERROR(
        HostTensorToMutableBorrowingLiteral(xla_shapes_[i], output, &literal));
    TF_RETURN_IF_ERROR(
        T::transfer_op_->TransferLiteralFromOutfeed(device_ordinal, literal));
  }
  return Status::OK();
}

// These ops execute on either the TPU device or the CPU device. When
// running on CPU they must specify a non-negative value for
// device_ordinal to indicate which TPU to receive outfeed from.
REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeue").Device(DEVICE_TPU_NODE).HostMemory("output"),
    StreamExecutorOutfeedDequeueOp<TpuTransferAsyncOpKernel>);
REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeue").Device(DEVICE_CPU),
    StreamExecutorOutfeedDequeueOp<TpuTransferAsyncOpKernel>);

REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeueTuple").Device(DEVICE_TPU_NODE).HostMemory("outputs"),
    StreamExecutorOutfeedDequeueTupleOp<TpuTransferAsyncOpKernel>);
REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeueTuple").Device(DEVICE_CPU),
    StreamExecutorOutfeedDequeueTupleOp<TpuTransferAsyncOpKernel>);

// Below ops take device_ordinal as an input tensor rather than a attribute.
REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeueV2").Device(DEVICE_TPU_NODE).HostMemory("output"),
    StreamExecutorOutfeedDequeueOp<TpuTransferAsyncDynamicOrdinalOpKernel>);
REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeueV2").Device(DEVICE_CPU),
    StreamExecutorOutfeedDequeueOp<TpuTransferAsyncDynamicOrdinalOpKernel>);

REGISTER_KERNEL_BUILDER(
    Name("OutfeedDequeueTupleV2").Device(DEVICE_TPU_NODE).HostMemory("outputs"),
    StreamExecutorOutfeedDequeueTupleOp<
        TpuTransferAsyncDynamicOrdinalOpKernel>);
REGISTER_KERNEL_BUILDER(Name("OutfeedDequeueTupleV2").Device(DEVICE_CPU),
                        StreamExecutorOutfeedDequeueTupleOp<
                            TpuTransferAsyncDynamicOrdinalOpKernel>);

}  // namespace tensorflow
