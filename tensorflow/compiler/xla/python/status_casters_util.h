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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_STATUS_CASTERS_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_STATUS_CASTERS_UTIL_H_

#include <optional>

#include "tensorflow/compiler/xla/status.h"

namespace xla {
namespace status_casters_util {

using FunctionPtr = void (*)(xla::Status);

// Sets the function pointer `fn` as payload in `status`. The function must
// accept `xla::Status` as a parameter, and its intended use is to cast it to a
// custom exception raised in Python code.
//
// Example:
//   void RaiseCustomException(xla::Status) {
//     throw MyCustomException("");
//   }
//   xla::Status status = ...;
//   SetFunctionPointerAsPayload(status, &RaiseCustomException);
void SetFunctionPointerAsPayload(xla::Status& status, FunctionPtr fn);

// Gets the function pointer from the `status` payload, returns std::nullopt if
// the function pointer was not set.
std::optional<FunctionPtr> GetFunctionPointerFromPayload(
    const xla::Status& status);

}  // namespace status_casters_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_STATUS_CASTERS_UTIL_H_
