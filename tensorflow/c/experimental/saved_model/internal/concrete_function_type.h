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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_INTERNAL_CONCRETE_FUNCTION_TYPE_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_INTERNAL_CONCRETE_FUNCTION_TYPE_H_

#include "tensorflow/c/experimental/saved_model/core/concrete_function.h"
#include "tensorflow/c/experimental/saved_model/internal/conversion_macros.h"

// Internal structures used by the SavedModel C API. These are likely to change
// and should not be depended on.

// It doesn't make sense to wrap tensorflow::ConcreteFunction* in a separate
// struct, since the lifetime of the struct and the raw pointer it wraps would
// be different. Therefore TF_ConcreteFunction* = tensorflow::ConcreteFunction*.
typedef struct TF_ConcreteFunction TF_ConcreteFunction;

namespace tensorflow {

DEFINE_CONVERSION_FUNCTIONS(tensorflow::ConcreteFunction, TF_ConcreteFunction)

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_INTERNAL_CONCRETE_FUNCTION_TYPE_H_
