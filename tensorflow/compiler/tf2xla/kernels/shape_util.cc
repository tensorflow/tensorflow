/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/shape_util.h"

#include <limits>

#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {

Status TensorShapeToConstant(const TensorShape& input_shape,
                             Tensor* shape_constant) {
  const int dims = input_shape.dims();
  if (shape_constant->dtype() == DT_INT32) {
    auto vec = shape_constant->vec<int32>();
    for (int i = 0; i < dims; ++i) {
      int64 dim_size = input_shape.dim_size(i);
      if (!FastBoundsCheck(dim_size, std::numeric_limits<int32>::max())) {
        return errors::InvalidArgument(
            "Shape with out_type=int32 does not support tensors > int32max",
            " but dim ", i, " is ", dim_size);
      }
      vec(i) = static_cast<int32>(dim_size);
    }
  } else {
    auto vec = shape_constant->vec<int64>();
    for (int i = 0; i < dims; ++i) {
      int64 dim_size = input_shape.dim_size(i);
      vec(i) = dim_size;
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
