/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_CUSTOM_CALL_REGISTRY_H_
#define XLA_SERVICE_GPU_RUNTIME_CUSTOM_CALL_REGISTRY_H_

#include <functional>

#include "xla/runtime/custom_call_registry.h"

namespace xla::gpu {

// This is a static custom call registry for XLA:GPU executables. XLA runtime
// custom calls must not be confused with a "classic" custom calls, they are
// an internal implementation of XLA runtime (and XLA:GPU by extension), and
// do not provide stable ABI across dynamically loaded libraries. XLA runtime
// custom calls must be statically linked.
//
// XLA:FFI is the planned mechanism for registering "custom calls" via a stable
// C ABI for internal and external uses, however it's under construction.
//
// See more XLA runtime and XLA FFI plans here:
// https://docs.google.com/document/d/1XHzJyfq-ZFn9WHoKe4o_urnwS991dFHgWoNRboBK_3I/edit#bookmark=id.696pyshem503
//
// XLA:FFI will become an official "external custom call" mechanism for XLA:GPU
// and XLA:CPU some time in 2024.

// Adds a direct custom call registration function to a static registry.
void AddDirectCustomCallRegistration(
    std::function<void(runtime::DirectCustomCallRegistry&)> registration);

// Registers all direct custom calls with the given registry.
void RegisterDirectCustomCalls(runtime::DirectCustomCallRegistry& registry);

//===----------------------------------------------------------------------===//
// Helper macro to define a static module registration.
//===----------------------------------------------------------------------===//

#define XLA_GPU_REGISTER_RUNTIME_CUSTOM_CALL(FUNC) \
  XLA_GPU_REGISTER_RUNTIME_CUSTOM_CALL_IMPL(FUNC, __COUNTER__)

#define XLA_GPU_REGISTER_RUNTIME_CUSTOM_CALL_IMPL(FUNC, N)           \
  static bool xla_gpu_runtime_custom_call_##N##_registered_ = []() { \
    ::xla::gpu::AddDirectCustomCallRegistration(FUNC);               \
    return true;                                                     \
  }()

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_CUSTOM_CALL_REGISTRY_H_
