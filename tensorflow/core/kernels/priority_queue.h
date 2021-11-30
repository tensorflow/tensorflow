/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_PRIORITY_QUEUE_H_
#define TENSORFLOW_CORE_KERNELS_PRIORITY_QUEUE_H_

#include <deque>
#include <queue>
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

using PriorityTensorPair = std::pair<int64_t, Tensor>;

struct ComparePriorityTensorPair {
  // 0 is a higher priority than 1, -MAX_LONG is a higher priority
  // than MAX_LONG, etc.  Values coming in with a smaller
  // priority number will bubble to the front of the queue.
  bool operator()(const PriorityTensorPair& lhs,
                  const PriorityTensorPair& rhs) const {
    return lhs.first > rhs.first;
  }
};

class PriorityQueue
    : public TypedQueue<std::priority_queue<PriorityTensorPair,
                                            std::vector<PriorityTensorPair>,
                                            ComparePriorityTensorPair> > {
 public:
  PriorityQueue(int32_t capacity, const DataTypeVector& component_dtypes,
                const std::vector<TensorShape>& component_shapes,
                const string& name);

  Status Initialize() override;  // Must be called before any other method.

  // Implementations of QueueInterface methods --------------------------------

  void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                  DoneCallback callback) override;
  void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                      DoneCallback callback) override;
  void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override;
  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      bool allow_small_batch,
                      CallbackWithTuple callback) override;
  Status MatchesNodeDef(const NodeDef& node_def) override;
  Status MatchesPriorityNodeDefTypes(const NodeDef& node_def) const;
  Status MatchesPriorityNodeDefShapes(const NodeDef& node_def) const;

  int32 size() const override {
    mutex_lock lock(mu_);
    return queues_[0].size();
  }

 private:
  ~PriorityQueue() override {}

  // Helper for dequeuing a single element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static Status GetElementComponentFromBatch(const Tuple& tuple, int index,
                                             int component,
                                             OpKernelContext* ctx,
                                             Tensor* out_element);

  TF_DISALLOW_COPY_AND_ASSIGN(PriorityQueue);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_PRIORITY_QUEUE_H_
