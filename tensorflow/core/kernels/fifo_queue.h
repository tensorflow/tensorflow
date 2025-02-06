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

#ifndef TENSORFLOW_CORE_KERNELS_FIFO_QUEUE_H_
#define TENSORFLOW_CORE_KERNELS_FIFO_QUEUE_H_

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/kernels/typed_queue.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class FIFOQueue : public TypedQueue<std::deque<Tensor> > {
 public:
  FIFOQueue(int32_t capacity, const DataTypeVector& component_dtypes,
            const std::vector<TensorShape>& component_shapes,
            const string& name);

  // Implementations of QueueInterface methods --------------------------------

  void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                  DoneCallback callback) override;
  void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                      DoneCallback callback) override;
  void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override;
  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      bool allow_small_batch,
                      CallbackWithTuple callback) override;
  absl::Status MatchesNodeDef(const NodeDef& node_def) override;

  int32 size() const override {
    mutex_lock lock(mu_);
    return queues_[0].size();
  }

 protected:
  ~FIFOQueue() override {}

  // Helper for dequeuing a single element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static absl::Status GetElementComponentFromBatch(const Tuple& tuple,
                                                   int64_t index, int component,
                                                   OpKernelContext* ctx,
                                                   Tensor* out_tensor);

 private:
  FIFOQueue(const FIFOQueue&) = delete;
  void operator=(const FIFOQueue&) = delete;
};

// Defines a FIFOQueueOp, which produces a Queue (specifically, one
// backed by FIFOQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
class FIFOQueueOp : public TypedQueueOp {
 public:
  explicit FIFOQueueOp(OpKernelConstruction* context);

 private:
  absl::Status CreateResource(QueueInterface** ret) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::vector<TensorShape> component_shapes_;
  FIFOQueueOp(const FIFOQueueOp&) = delete;
  void operator=(const FIFOQueueOp&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FIFO_QUEUE_H_
