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

#include "tensorflow/c/tf_shape.h"

#include <stdint.h>

#include "tensorflow/c/tf_shape_internal.h"
#include "tensorflow/core/framework/tensor_shape.h"

extern "C" {

TF_Shape* TF_NewShape() {
  return tensorflow::wrap(new tensorflow::PartialTensorShape());
}

int TF_ShapeDims(const TF_Shape* shape) {
  return tensorflow::unwrap(shape)->dims();
}

int64_t TF_ShapeDimSize(const TF_Shape* shape, int d) {
  return tensorflow::unwrap(shape)->dim_size(d);
}

void TF_DeleteShape(TF_Shape* shape) { delete tensorflow::unwrap(shape); }

}  // end extern "C"
