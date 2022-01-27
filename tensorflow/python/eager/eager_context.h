/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_EAGER_EAGER_CONTEXT_H_
#define TENSORFLOW_PYTHON_EAGER_EAGER_CONTEXT_H_

#include "tensorflow/c/eager/c_api.h"

namespace tensorflow {
namespace eager {

// Sets the EagerContext owned by the current Python eager Context (see
// TFE_Py_SetEagerContext in pywrap_tfe.h). This is always called in tandem with
// TFE_Py_SetEagerContext (but not called by it, because its py_context
// argument is opaque).
//
// Do not use this function in production. It is only intended for testing.
// (see _reset_context in context.py).
//
// Not thread-safe.
void TFE_Py_SetCEagerContext(TFE_Context* ctx);

// Returns the EagerContext owned by the current Python eager Context (see
// TFE_Py_SetEagerContext in pywrap_tfe.h).
//
// Not thread-safe.
TFE_Context* GetCEagerContext();

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_EAGER_EAGER_CONTEXT_H_
