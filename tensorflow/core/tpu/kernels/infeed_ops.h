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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_INFEED_OPS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_INFEED_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/kernels/transfer_ops.h"

namespace tensorflow {

// TODO(b/65200690): Rework this when there is a callback based infeed API to
// StreamExecutor.

// The InfeedEnqueue op is used to deliver data to the device infeed queue.
class TpuInfeedEnqueueOp : public TpuTransferAsyncOpKernel {
 public:
  explicit TpuInfeedEnqueueOp(
      OpKernelConstruction* ctx,
      std::unique_ptr<TpuTransferOpInterface> transfer_op);
  Status DoWork(OpKernelContext* ctx, int device_ordinal) override;

 private:
  TensorShape shape_;
  DataType dtype_;
  xla::Shape xla_shape_;

  // TpuInfeedEnqueueOp is neither copyable nor movable.
  TpuInfeedEnqueueOp(const TpuInfeedEnqueueOp&) = delete;
  TpuInfeedEnqueueOp& operator=(const TpuInfeedEnqueueOp&) = delete;
};

// The InfeedEnqueueTuple op is used on the host to deliver multiple tensors to
// the device infeed queue as an XLA tuple.
class TpuInfeedEnqueueTupleOp : public TpuTransferAsyncOpKernel {
 public:
  explicit TpuInfeedEnqueueTupleOp(
      OpKernelConstruction* ctx,
      std::unique_ptr<TpuTransferOpInterface> transfer_op);
  Status DoWork(OpKernelContext* ctx, int device_ordinal) override;

 private:
  std::vector<TensorShape> shapes_;
  DataTypeVector dtypes_;
  xla::Shape tuple_shape_;

  // TpuInfeedEnqueueTupleOp is neither copyable nor movable.
  TpuInfeedEnqueueTupleOp(const TpuInfeedEnqueueTupleOp&) = delete;
  TpuInfeedEnqueueTupleOp& operator=(const TpuInfeedEnqueueTupleOp&) = delete;
};

// The InfeedEnqueuePrelinearizedBufferOp op is used to transfer prelinearized
// buffers to the device infeed queue.
class InfeedEnqueuePrelinearizedBufferOp : public TpuTransferAsyncOpKernel {
 public:
  explicit InfeedEnqueuePrelinearizedBufferOp(
      OpKernelConstruction* ctx,
      std::unique_ptr<TpuTransferOpInterface> transfer_op);

  Status DoWork(OpKernelContext* ctx, int device_ordinal) override;

 private:
  // InfeedEnqueuePrelinearizedBufferOp is neither copyable nor movable.
  InfeedEnqueuePrelinearizedBufferOp(
      const InfeedEnqueuePrelinearizedBufferOp&) = delete;
  InfeedEnqueuePrelinearizedBufferOp& operator=(
      const InfeedEnqueuePrelinearizedBufferOp&) = delete;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_INFEED_OPS_H_
