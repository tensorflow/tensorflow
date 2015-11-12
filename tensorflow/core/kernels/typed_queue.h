#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_TYPED_QUEUE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_TYPED_QUEUE_H_

#include <vector>

#include "tensorflow/core/kernels/queue_base.h"

namespace tensorflow {

// TypedQueue builds on QueueBase, with backing class (SubQueue)
// known and stored within.  Shared methods that need to have access
// to the backed data sit in this class.
template <typename SubQueue>
class TypedQueue : public QueueBase {
 public:
  TypedQueue(const int32 capacity, const DataTypeVector& component_dtypes,
             const std::vector<TensorShape>& component_shapes,
             const string& name);

  virtual Status Initialize();  // Must be called before any other method.

 protected:
  std::vector<SubQueue> queues_ GUARDED_BY(mu_);
};  // class TypedQueue

template <typename SubQueue>
TypedQueue<SubQueue>::TypedQueue(
    int32 capacity, const DataTypeVector& component_dtypes,
    const std::vector<TensorShape>& component_shapes, const string& name)
    : QueueBase(capacity, component_dtypes, component_shapes, name) {}

template <typename SubQueue>
Status TypedQueue<SubQueue>::Initialize() {
  if (component_dtypes_.empty()) {
    return errors::InvalidArgument("Empty component types for queue ", name_);
  }
  if (!component_shapes_.empty() &&
      component_dtypes_.size() != component_shapes_.size()) {
    return errors::InvalidArgument("Different number of component types (",
                                   component_dtypes_.size(), ") vs. shapes (",
                                   component_shapes_.size(), ").");
  }

  mutex_lock lock(mu_);
  queues_.reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    queues_.push_back(SubQueue());
  }
  return Status::OK();
}

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_TYPED_QUEUE_H_
