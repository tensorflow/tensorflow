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
#include "tensorflow/c/eager/tracing_utils.h"

#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/experimental/gradients/tape/tape_operation.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace tracing {

absl::Status MaybeSetOpName(AbstractOperation* op, const char* op_name) {
  if (isa<TracingOperation>(op)) {
    TF_RETURN_IF_ERROR(dyn_cast<TracingOperation>(op)->SetOpName(op_name));
  }
  if (isa<gradients::TapeOperation>(op)) {
    TF_RETURN_IF_ERROR(MaybeSetOpName(
        dyn_cast<gradients::TapeOperation>(op)->GetBackingOperation(),
        op_name));
  }
  return absl::OkStatus();
}
}  // namespace tracing
}  // namespace tensorflow
