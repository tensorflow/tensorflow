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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tpu/kernels/transfer_ops.h"

namespace tensorflow {

// The OutfeedDequeue op is used to retrieve a single tensor from the device
// outfeed queue.
template <class T>
class TpuOutfeedDequeueOp : public T {
 public:
  explicit TpuOutfeedDequeueOp(OpKernelConstruction* ctx);

  Status DoWork(OpKernelContext* ctx,
                xla::TpuTransferManagerInterface* transfer_manager,
                stream_executor::StreamExecutor* stream_executor) override;

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
  explicit TpuOutfeedDequeueTupleOp(OpKernelConstruction* ctx);

  Status DoWork(OpKernelContext* ctx,
                xla::TpuTransferManagerInterface* transfer_manager,
                stream_executor::StreamExecutor* stream_executor) override;

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
