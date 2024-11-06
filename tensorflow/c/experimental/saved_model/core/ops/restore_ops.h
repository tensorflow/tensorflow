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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_OPS_RESTORE_OPS_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_OPS_RESTORE_OPS_H_

#include <string>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace internal {

// TODO(bmzhao): Add a function to restore multiple tensors in one call.

// Restores a single non-partioned tensorhandle of dtype `dtype`, using
// checkpoint at `prefix`, with a value stored in `checkpoint_key`.
absl::Status SingleRestore(ImmediateExecutionContext* ctx,
                           const std::string& prefix,
                           const std::string& checkpoint_key, DataType dtype,
                           ImmediateTensorHandlePtr* out);

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_OPS_RESTORE_OPS_H_
