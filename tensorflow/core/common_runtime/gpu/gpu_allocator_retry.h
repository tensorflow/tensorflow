#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ALLOCATOR_RETRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ALLOCATOR_RETRY_H_

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/env.h"

namespace tensorflow {

// A retrying wrapper for a memory allocator.
class GPUAllocatorRetry {
 public:
  GPUAllocatorRetry();

  // Call 'alloc_func' to obtain memory.  On first call,
  // 'verbose_failure' will be false.  If return value is nullptr,
  // then wait up to 'max_millis_to_wait' milliseconds, retrying each
  // time a call to DeallocateRaw() is detected, until either a good
  // pointer is returned or the deadline is exhausted.  If the
  // deadline is exahusted, try one more time with 'verbose_failure'
  // set to true.  The value returned is either the first good pointer
  // obtained from 'alloc_func' or nullptr.
  void* AllocateRaw(std::function<void*(size_t alignment, size_t num_bytes,
                                        bool verbose_failure)> alloc_func,
                    int max_millis_to_wait, size_t alignment, size_t bytes);

  // Calls dealloc_func(ptr) and then notifies any threads blocked in
  // AllocateRaw() that would like to retry.
  void DeallocateRaw(std::function<void(void* ptr)> dealloc_func, void* ptr);

 private:
  Env* env_;
  mutex mu_;
  condition_variable memory_returned_;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ALLOCATOR_RETRY_H_
