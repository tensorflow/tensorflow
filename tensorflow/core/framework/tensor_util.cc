#include "tensorflow/core/framework/tensor_util.h"

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
namespace tensor {

Tensor DeepCopy(const Tensor& other) {
  Tensor tmp = Tensor(other.dtype(), other.shape());
  if (DataTypeCanUseMemcpy(other.dtype())) {
    StringPiece other_data = other.tensor_data();

    // We use StringPiece as a convenient map over the tensor buffer,
    // but we cast the type to get to the underlying buffer to do the
    // copy.
    StringPiece tmp_data = tmp.tensor_data();
    memcpy(const_cast<char*>(tmp_data.data()), other_data.data(),
           other_data.size());
  } else {
    CHECK_EQ(DT_STRING, other.dtype());
    tmp.flat<string>() = other.flat<string>();
  }
  return tmp;
}

}  // namespace tensor
}  // namespace tensorflow
