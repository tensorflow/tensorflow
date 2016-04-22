#ifndef TENSORFLOW_CORE_KERNELS_BUCKETED_TYPED_QUEUE_H_
#define TENSORFLOW_CORE_KERNELS_BUCKETED_TYPED_QUEUE_H_

#include <unordered_map>
#include <vector>

#include "tensorflow/core/kernels/typed_queue.h"

namespace tensorflow {

template <typename SubQueue>
class BucketedTypedQueue : public TypedQueue<SubQueue> {
 public:
  BucketedTypedQueue(
      const int buckets, const int32 capacity,
      const DataTypeVector& component_dtypes,
      const std::vector<TensorShape>& component_shapes, const string& name);

  virtual Status Initialize();  // Must be called before any other method.

 protected:
  int buckets_;
  std::unordered_map<int, std::vector<SubQueue>> bucketed_queues_
      GUARDED_BY(mu_);
};  // class BucketedTypedQueue

template <typename SubQueue>
BucketedTypedQueue<SubQueue>::BucketedTypedQueue(
    int buckets, int32 capacity, const DataTypeVector& component_dtypes,
    const std::vector<TensorShape>& component_shapes, const string& name)
    : TypedQueue<SubQueue>(capacity, component_dtypes, component_shapes, name),
                           buckets_(buckets) {}

template <typename SubQueue>
Status BucketedTypedQueue<SubQueue>::Initialize() {
  Status s = TypedQueue<SubQueue>::Initialize();
  if (!s.ok()) return s;

  mutex_lock lock(this->mu_);
  for (int b = 0; b < buckets_; ++b) {
    auto& queues = bucketed_queues_[b];
    queues.reserve(this->num_components());
    for (int i = 0; i < this->num_components(); ++i) {
      queues.push_back(SubQueue());
    }
  }
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BUCKETED_TYPED_QUEUE_H_
