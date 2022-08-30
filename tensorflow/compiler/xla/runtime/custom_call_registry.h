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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_CUSTOM_CALL_REGISTRY_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_CUSTOM_CALL_REGISTRY_H_

#include <memory>
#include <string_view>

#include "tensorflow/compiler/xla/runtime/custom_call.h"

namespace xla {
namespace runtime {

// Custom call registry is a container for the custom calls that looks up the
// handler implementing the custom call by name at run time. It is used to
// implement a generic `rt.custom_call` runtime intrinsic.
//
// For low overhead custom calls prefer direct custom calls that linked with the
// compiled executable and bypass by-name look up (see DirectCustomCallLibrary).
//
// TODO(ezhulenev): Consider removing this registry, because we'll likely not
// need it for any of the practical purposes, and it's currently used only in
// tests. We also likely don't need the generic custom call API.
class CustomCallRegistry {
 public:
  // The type for custom call registration functions.
  using RegistrationFunction = void (*)(CustomCallRegistry*);

  CustomCallRegistry();
  ~CustomCallRegistry() = default;

  CustomCallRegistry(const CustomCallRegistry&) = delete;
  CustomCallRegistry& operator=(const CustomCallRegistry&) = delete;

  void Register(std::unique_ptr<class CustomCall> custom_call);

  class CustomCall* Find(std::string_view callee) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// TODO(ezhulenev): Remove static registration mechanism, and pass custom call
// registry explicitly to the JitExecutable. It should be passed around as a
// part of XLA runtime execution context.

// Use this macro to add a function that will register custom calls that are
// statically linked in the binary. FUNC should be a function pointer with the
// prototype given by the CustomCallRegistry::RegistrationFunction alias.
#define XLA_RUNTIME_STATIC_CUSTOM_CALL_REGISTRATION(FUNC) \
  XLA_RUNTIME_STATIC_CUSTOM_CALL_REGISTRATION_IMPL(FUNC, __COUNTER__)
#define XLA_RUNTIME_STATIC_CUSTOM_CALL_REGISTRATION_IMPL(FUNC, N) \
  XLA_RUNTIME_STATIC_CUSTOM_CALL_REGISTRATION_IMPL_EXPAND(FUNC, N)
#define XLA_RUNTIME_STATIC_CUSTOM_CALL_REGISTRATION_IMPL_EXPAND(FUNC, N) \
  static bool XLA_RUNTIME_static_custom_call_##N##_registered_ = []() {  \
    ::xla::runtime::AddStaticCustomCallRegistration(FUNC);               \
    return true;                                                         \
  }()

// Registers all statically linked custom calls in the given registry.
void RegisterStaticCustomCalls(CustomCallRegistry* custom_call_registry);

// Adds a custom call registration function to the registry. This should not be
// used directly; use XLA_RUNTIME_STATIC_CUSTOM_CALL_REGISTRATION instead.
void AddStaticCustomCallRegistration(
    CustomCallRegistry::RegistrationFunction registration);

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_CUSTOM_CALL_REGISTRY_H_
