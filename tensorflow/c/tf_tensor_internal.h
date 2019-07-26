/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_C_TF_TENSOR_INTERNAL_H_
#define TENSORFLOW_C_TF_TENSOR_INTERNAL_H_

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

// Internal structures used by the C API. These are likely to change and should
// not be depended on.

struct TF_Tensor {
  ~TF_Tensor();

  TF_DataType dtype;
  tensorflow::TensorShape shape;
  tensorflow::TensorBuffer* buffer;
};

namespace tensorflow {

class TensorCApi {
 public:
  static TensorBuffer* Buffer(const Tensor& tensor) { return tensor.buf_; }
  static Tensor MakeTensor(TF_DataType type, const TensorShape& shape,
                           TensorBuffer* buf) {
    return Tensor(static_cast<DataType>(type), shape, buf);
  }
};

}  // namespace tensorflow
#endif  // TENSORFLOW_C_TF_TENSOR_INTERNAL_H_
