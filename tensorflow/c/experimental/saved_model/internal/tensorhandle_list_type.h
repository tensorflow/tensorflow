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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_INTERNAL_CONCRETE_FUNCTION_LIST_TYPE_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_INTERNAL_CONCRETE_FUNCTION_LIST_TYPE_H_

#include <vector>

#include "tensorflow/c/conversion_macros.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"

// Internal structures used by the SavedModel C API. These are likely to
// change and should not be depended on.

typedef struct TF_TensorHandleList TF_TensorHandleList;

namespace tensorflow {

DEFINE_CONVERSION_FUNCTIONS(
    std::vector<tensorflow::ImmediateExecutionTensorHandle*>,
    TF_TensorHandleList)

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_INTERNAL_CONCRETE_FUNCTION_LIST_TYPE_H_
