#include "tensorflow/core/framework/tensor_reference.h"

namespace tensorflow {

TensorReference::TensorReference(const Tensor& tensor)
    : buf_(tensor.buf_ ? tensor.buf_->root_buffer() : nullptr) {
  if (buf_) buf_->Ref();
}

}  // namespace tensorflow
