#ifndef TENSORFLOW_FRAMEWORK_TENSOR_UTIL_H_
#define TENSORFLOW_FRAMEWORK_TENSOR_UTIL_H_

#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
namespace tensor {

// DeepCopy returns a tensor whose contents are a deep copy of the
// contents of 'other'.  This function is intended only for
// convenience, not speed.
//
// REQUIRES: 'other' must point to data stored in CPU memory.
// REQUIRES: 'other' must be a Tensor of a copy-able type if
//           'other' is not appropriately memory-aligned.
Tensor DeepCopy(const Tensor& other);

}  // namespace tensor
}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_TENSOR_UTIL_H_
