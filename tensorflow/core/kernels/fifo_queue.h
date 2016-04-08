/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_FIFO_QUEUE_H_
#define TENSORFLOW_KERNELS_FIFO_QUEUE_H_

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/typed_queue.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class FIFOQueue : public TypedQueue<std::deque<PersistentTensor> > {
 public:
  FIFOQueue(int32 capacity, const DataTypeVector& component_dtypes,
            const std::vector<TensorShape>& component_shapes,
            const string& name);

  // Implementations of QueueInterface methods --------------------------------

  void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                  DoneCallback callback) override;
  void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                      DoneCallback callback) override;
  void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override;
  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      CallbackWithTuple callback) override;
  Status MatchesNodeDef(const NodeDef& node_def) override;

  int32 size() override {
    mutex_lock lock(mu_);
    return queues_[0].size();
  }

 protected:
  ~FIFOQueue() override {}

  // Helper for dequeuing a single element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static Status GetElementComponentFromBatch(const Tuple& tuple, int64 index,
                                             int component,
                                             OpKernelContext* ctx,
                                             PersistentTensor* out_element);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FIFOQueue);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FIFO_QUEUE_H_
