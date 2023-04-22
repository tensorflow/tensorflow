/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CUSTOM_CALL_TARGET_REGISTRY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CUSTOM_CALL_TARGET_REGISTRY_H_

// This file is depended on by kernels that have to build for mobile devices.
// For this reason, we avoid relying on TensorFlow and instead only use the
// standard C++ library.

#include <map>
#include <mutex>  // NOLINT
#include <string>

namespace xla {

// XLA JIT compilers use this registry to resolve symbolic CustomCall targets;
// so when using XLA as a JIT, CustomCall targets need to be registered here
// with the symbol name used in the CustomCall.
//
// The XLA:CPU ahead-of-time (AOT) compiler links using a standard offline
// linker; so when compiling in CPU AOT mode, you *also* need to make sure the
// name of the callee (presumably implemented in C++) matches up with the
// symbolic name used in the CustomCall.
//
// We maintain the registry in both the JIT and the AOT cases for simplicity,
// but we only use it when running in JIT mode.
class CustomCallTargetRegistry {
 public:
  static CustomCallTargetRegistry* Global();

  void Register(const std::string& symbol, void* address,
                const std::string& platform);
  void* Lookup(const std::string& symbol, const std::string& platform) const;

 private:
  // Maps the pair (symbol, platform) to a C function implementing a custom call
  // named `symbol` for StreamExecutor platform `platform`.
  //
  // Different platforms have different ABIs.  TODO(jlebar): Describe them!
  //
  // (We std::map rather than std::unordered_map because the STL doesn't provide
  // a default hasher for pair<string, string>, and we want to avoid pulling in
  // dependencies that might define this.)
  std::map<std::pair<std::string, std::string>, void*> registered_symbols_;
  mutable std::mutex mu_;
};

class RegisterCustomCallTarget {
 public:
  explicit RegisterCustomCallTarget(const std::string& name, void* address,
                                    const std::string& platform) {
    CustomCallTargetRegistry::Global()->Register(name, address, platform);
  }
};

#define XLA_REGISTER_CUSTOM_CALL_CONCAT(a, b) a##b

#define XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM_HELPER(symbol, address,   \
                                                        platform, counter) \
  static ::xla::RegisterCustomCallTarget XLA_REGISTER_CUSTOM_CALL_CONCAT(  \
      custom_call_target_register, counter)(                               \
      symbol, reinterpret_cast<void*>(address), platform)

#define XLA_REGISTER_CUSTOM_CALL_TARGET(function, platform) \
  XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(#function, function, platform)

#define XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address, platform)  \
  XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM_HELPER(symbol, address, platform, \
                                                  __COUNTER__)

// Convenience overloads for registering custom-call targets on the CPU.
#define XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(function) \
  XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(#function, function, "Host")

#define XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address) \
  XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address, "Host")

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CUSTOM_CALL_TARGET_REGISTRY_H_
