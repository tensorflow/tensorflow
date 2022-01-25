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

#include "tensorflow/python/eager/eager_context.h"

#include "tensorflow/c/eager/c_api.h"

namespace tensorflow {
namespace eager {

namespace {
// This object tracks the EagerContext owned by global_py_eager_context in
// pywrap_tfe_src.cc. Since the vast majority of the Python API is dependent on
// that global_py_eager_context (including memory management), the Py object
// owns the C object, so this pointer is non-owning.
TFE_Context* global_c_eager_context = nullptr;
}  // namespace

void TFE_Py_SetCEagerContext(TFE_Context* ctx) { global_c_eager_context = ctx; }

TFE_Context* GetCEagerContext() { return global_c_eager_context; }

}  // namespace eager
}  // namespace tensorflow
