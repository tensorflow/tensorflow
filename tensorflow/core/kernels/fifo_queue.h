#ifndef TENSORFLOW_KERNELS_FIFO_QUEUE_H_
#define TENSORFLOW_KERNELS_FIFO_QUEUE_H_

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/typed_queue.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

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

 private:
  ~FIFOQueue() override {}

  // Helper for dequeuing a single element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static Status GetElementComponentFromBatch(const Tuple& tuple, int index,
                                             int component,
                                             OpKernelContext* ctx,
                                             PersistentTensor* out_element);

  TF_DISALLOW_COPY_AND_ASSIGN(FIFOQueue);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FIFO_QUEUE_H_
