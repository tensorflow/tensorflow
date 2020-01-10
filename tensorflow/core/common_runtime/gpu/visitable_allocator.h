#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_VISITABLE_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_VISITABLE_ALLOCATOR_H_

#include <functional>
#include "tensorflow/core/framework/allocator.h"

namespace tensorflow {

// Subclass VisitableAllocator instead of Allocator when a memory
// allocator needs to enable some kind of registration/deregistration
// of memory areas.
class VisitableAllocator : public Allocator {
 public:
  // Visitor gets called with a pointer to a memory area and its
  // size in bytes.
  typedef std::function<void(void*, size_t)> Visitor;

  // Register a visitor guaranteed to be called exactly once on each
  // chunk of memory newly allocated from the underlying device.
  // Typically, chunks will be reused and possibly sub-divided by a
  // pool manager, so the calls will happen only once per process
  // execution, not once per tensor (re)allocation.
  virtual void AddAllocVisitor(Visitor visitor) = 0;

  // Register a visitor guaranteed to be called on each chunk of
  // memory returned to the underlying device.
  virtual void AddFreeVisitor(Visitor visitor) = 0;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_VISITABLE_ALLOCATOR_H_
