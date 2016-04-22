#ifndef TENSORFLOW_KERNELS_FIFO_BUCKETED_QUEUE_H_
#define TENSORFLOW_KERNELS_FIFO_BUCKETED_QUEUE_H_

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bucketed_typed_queue.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class FIFOBucketedQueue
  : public BucketedTypedQueue<std::deque<PersistentTensor>> {
 public:
  FIFOBucketedQueue(
      int buckets, int capacity, int batch_size,
      const DataTypeVector& component_dtypes,
      const std::vector<TensorShape>& component_shapes, const string& name);

  // Implementations of QueueInterface methods --------------------------------

  void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                  DoneCallback callback) override;
  void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                      DoneCallback callback) override;
  void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override;
  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      CallbackWithTuple callback) override;
  Status MatchesNodeDef(const NodeDef& node_def) override;

  void BatchBucketedQueuesToQueuesLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  int32 size() override {
    return size_;
  }

 protected:
  ~FIFOBucketedQueue() override {}

  // Helper for dequeuing a single element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static Status GetElementComponentFromBatch(const Tuple& tuple, int64 index,
                                             int component,
                                             OpKernelContext* ctx,
                                             PersistentTensor* out_element);

 private:
  int batch_size_;
  int32 size_ = 0 GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(FIFOBucketedQueue);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FIFO_BUCKETED_QUEUE_H_
