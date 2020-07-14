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

#ifndef TENSORFLOW_C_TENSOR_SHAPE_UTILS_H_
#define TENSORFLOW_C_TENSOR_SHAPE_UTILS_H_

#include <string>

#include "tensorflow/c/tf_tensor.h"

// The following are utils for the shape of a TF_Tensor type. 
// These functions may later be subsumed by the methods for a
// TF_TensorShape type 

// Returns a string representation of the TF_Tensor
std::string TF_ShapeDebugString(TF_Tensor* tensor); 

#endif  // TENSORFLOW_C_TENSOR_SHAPE_UTILS_H_

