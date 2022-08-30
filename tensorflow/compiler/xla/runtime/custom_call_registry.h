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

#include "llvm/ADT/StringMap.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"

namespace xla {
namespace runtime {

// Custom call registry is a container for the custom calls that looks up the
// handler implementing the custom call by name at run time. It is used to
// implement a generic `rt.custom_call` runtime intrinsic.
//
// For low overhead custom calls prefer direct custom calls that linked with the
// compiled executable and bypass by-name look up (see DirectCustomCallLibrary).
class CustomCallRegistry {
 public:
  // The type for custom call registration functions.
  using RegistrationFunction = void (*)(CustomCallRegistry*);

  CustomCallRegistry() = default;

  void Register(std::unique_ptr<class CustomCall> custom_call);

  class CustomCall* Find(std::string_view callee) const;

 private:
  llvm::StringMap<std::unique_ptr<CustomCall>> custom_calls_;
};

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_CUSTOM_CALL_REGISTRY_H_
