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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_H_

#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"

namespace xla {
namespace runtime {
namespace ffi {

// Export FFI functions registered with a runtime as a dynamic custom call
// registry, so it can be passed to the XLA runtime executable.
DynamicCustomCallRegistry& FfiCustomCalls();

}  // namespace ffi
}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_H_
