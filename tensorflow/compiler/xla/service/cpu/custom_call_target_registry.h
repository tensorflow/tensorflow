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
#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CUSTOM_CALL_TARGET_REGISTRY_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CUSTOM_CALL_TARGET_REGISTRY_H_

// This file is depended on by kernels that have to build for mobile devices.
// For this reason, we avoid relying on TensorFlow and instead only use the
// standard C++ library.

#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>

namespace xla {
namespace cpu {

// The CPU JIT compiler uses this registry to resolve symbolic CustomCall
// targets; so when using the CPU JIT, CustomCall targets need to be registered
// here with the symbol name used in the CustomCall.
//
// The XLA AOT compiler links using a standard offline linker; so when compiling
// in AOT mode, you *also* need to make sure the name of the callee (presumably
// implemented in C++) matches up with the symbolic name used in the CustomCall.
//
// We maintain the registry in both the JIT and the AOT cases for simplicity,
// but we only use it when running in JIT mode.
class CustomCallTargetRegistry {
 public:
  static CustomCallTargetRegistry* Global();

  void Register(const std::string& symbol, void* address);
  void* Lookup(const std::string& symbol) const;

 private:
  std::unordered_map<std::string, void*> registered_symbols_;
  mutable std::mutex mu_;
};

class RegisterCustomCallTarget {
 public:
  explicit RegisterCustomCallTarget(const std::string& name, void* address) {
    CustomCallTargetRegistry::Global()->Register(name, address);
  }
};

#define REGISTER_CUSTOM_CALL_CONCAT(a, b) a##b

#define REGISTER_CUSTOM_CALL_TARGET_WITH_SYM_HELPER(symbol, address, counter) \
  static ::xla::cpu::RegisterCustomCallTarget REGISTER_CUSTOM_CALL_CONCAT(    \
      custom_call_target_register, counter)(symbol,                           \
                                            reinterpret_cast<void*>(address))

#define REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address) \
  REGISTER_CUSTOM_CALL_TARGET_WITH_SYM_HELPER(symbol, address, __COUNTER__)

#define REGISTER_CUSTOM_CALL_TARGET(function) \
  REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(#function, function)

}  // namespace cpu
}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CUSTOM_CALL_TARGET_REGISTRY_H_
