#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_DMA_HELPER_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_DMA_HELPER_H_

#include "tensorflow/core/public/tensor.h"

// For internal use only.  Visibility should be limited to brain/framework.

namespace tensorflow {
class DMAHelper {
 public:
  static bool CanUseDMA(const Tensor* t) { return t->CanUseDMA(); }
  static const void* base(const Tensor* t) { return t->base<const void>(); }
  static void* base(Tensor* t) { return t->base<void>(); }
  static TensorBuffer* buffer(Tensor* t) { return t->buf_; }
  static const TensorBuffer* buffer(const Tensor* t) { return t->buf_; }
};
}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_DMA_HELPER_H_
