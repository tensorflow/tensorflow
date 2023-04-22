/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/kernels/tensor_shape_utils.h"

#include <string>

#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

std::string ShapeDebugString(TF_Tensor* tensor) {
  // A TF_Tensor cannot have an unknown rank.
  CHECK_GE(TF_NumDims(tensor), 0);
  tensorflow::string s = "[";
  for (int i = 0; i < TF_NumDims(tensor); ++i) {
    if (i > 0) tensorflow::strings::StrAppend(&s, ",");
    int64_t dim = TF_Dim(tensor, i);
    // A TF_Tensor cannot have an unknown dimension.
    CHECK_GE(dim, 0);
    tensorflow::strings::StrAppend(&s, dim);
  }
  tensorflow::strings::StrAppend(&s, "]");
  return s;
}
}  // namespace tensorflow
