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
#ifndef TENSORFLOW_C_EAGER_TFE_CONTEXT_INTERNAL_H_
#define TENSORFLOW_C_EAGER_TFE_CONTEXT_INTERNAL_H_

#include "tensorflow/c/conversion_macros.h"
#include "tensorflow/c/eager/immediate_execution_context.h"

// Wraps a pointer to a context implementation.
//
// WARNING: Since the underlying object could be ref-counted a user of this
// interface cannot destruct the underlying context object. Instead, call
// TFE_DeleteContext who calls Release() on the context pointer and deletes
// the TFE_Context structure.
typedef struct TFE_Context TFE_Context;

namespace tensorflow {

DEFINE_CONVERSION_FUNCTIONS(tensorflow::ImmediateExecutionContext, TFE_Context);

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_TFE_CONTEXT_INTERNAL_H_
