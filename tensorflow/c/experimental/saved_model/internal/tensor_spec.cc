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

#include "tensorflow/c/experimental/saved_model/public/tensor_spec.h"

#include "tensorflow/c/experimental/saved_model/core/tensor_spec.h"
#include "tensorflow/c/experimental/saved_model/internal/tensor_spec_type.h"
#include "tensorflow/c/tf_shape_internal.h"

extern "C" {

TF_DataType TF_TensorSpecDataType(const TF_TensorSpec* spec) {
  return static_cast<TF_DataType>(tensorflow::unwrap(spec)->dtype());
}

const TF_Shape* TF_TensorSpecShape(const TF_TensorSpec* spec) {
  return tensorflow::wrap(&tensorflow::unwrap(spec)->shape());
}

}  // end extern "C"
