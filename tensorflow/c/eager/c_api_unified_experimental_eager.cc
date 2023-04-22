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

#include <vector>

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/strcat.h"

// =============================================================================
// Public C API entry points
// These are only the entry points specific to the Eager API.
// =============================================================================

using tensorflow::AbstractContext;
using tensorflow::AbstractTensorHandle;
using tensorflow::dyn_cast;
using tensorflow::ImmediateExecutionContext;
using tensorflow::ImmediateExecutionTensorHandle;
using tensorflow::string;
using tensorflow::unwrap;
using tensorflow::wrap;
using tensorflow::strings::StrCat;

TF_ExecutionContext* TF_NewEagerExecutionContext(TFE_ContextOptions* options,
                                                 TF_Status* s) {
  TFE_Context* c_ctx = TFE_NewContext(options, s);
  if (TF_GetCode(s) != TF_OK) {
    return nullptr;
  }
  return wrap(static_cast<AbstractContext*>(unwrap(c_ctx)));
}

TF_AbstractTensor* TF_CreateAbstractTensorFromEagerTensor(TFE_TensorHandle* t,
                                                          TF_Status* s) {
  return wrap(static_cast<AbstractTensorHandle*>(unwrap(t)));
}

TFE_TensorHandle* TF_AbstractTensorGetEagerTensor(TF_AbstractTensor* at,
                                                  TF_Status* s) {
  auto handle = dyn_cast<ImmediateExecutionTensorHandle>(unwrap(at));
  if (!handle) {
    string msg =
        StrCat("Not an eager tensor handle.", reinterpret_cast<uintptr_t>(at));
    TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
    return nullptr;
  }
  return wrap(handle);
}

TFE_Context* TF_ExecutionContextGetTFEContext(TF_ExecutionContext* ctx,
                                              TF_Status* s) {
  auto imm_ctx = dyn_cast<ImmediateExecutionContext>(unwrap(ctx));
  if (!imm_ctx) {
    string msg =
        StrCat("Not an eager context.", reinterpret_cast<uintptr_t>(ctx));
    TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
    return nullptr;
  }
  return wrap(imm_ctx);
}
