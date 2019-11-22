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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_VARIABLE_RESOURCE_VARIABLE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_VARIABLE_RESOURCE_VARIABLE_H_

#include <unordered_map>

#include "tensorflow/lite/c/c_api_internal.h"

namespace tflite {

/// WARNING: Experimental interface, subject to change.
// A resource variable class. It's similar to TensorFlow Resource
// Variable, but it's identified with int32 ID in TFLite (instead of
// using Resource handle like TensorFlow).
//
// TODO(b/137042749): TFLite converter cannot convert variables yet.
// Variable functionalities are only tested with unit tests now.
class ResourceVariable {
 public:
  ResourceVariable();
  ResourceVariable(ResourceVariable&& other);

  ResourceVariable(const ResourceVariable&) = delete;
  ResourceVariable& operator=(const ResourceVariable&) = delete;

  ~ResourceVariable();

  // Assigns data from a tensor. Copies its type, shape and data over.
  TfLiteStatus AssignFrom(const TfLiteTensor* tensor);

  // Get the data tensor stored in the resource variable.
  // Returns `nullptr` if the variable is never initialized by calling
  // `AssignFrom`.
  TfLiteTensor* GetTensor() { return is_initialized_ ? &tensor_ : nullptr; }

 private:
  // The tensor (and its buffer stored in `tensor_.data` is fully owned by
  // the `ResourceVariable` object.
  TfLiteTensor tensor_;
  // True if `AssignFrom` function is every called.
  // False if and only if `tensor_` is filled with zeros.
  bool is_initialized_ = false;
};

using ResourceVariableMap = std::unordered_map<int, ResourceVariable>;

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_VARIABLE_RESOURCE_VARIABLE_H_
