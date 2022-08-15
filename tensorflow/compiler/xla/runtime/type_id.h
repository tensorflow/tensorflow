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

#ifndef XLA_RUNTIME_TYPE_ID_H_
#define XLA_RUNTIME_TYPE_ID_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/TypeID.h"  // from @llvm-project

namespace xla {
namespace runtime {

using ::mlir::TypeID;  // NOLINT

class TypeIDNameRegistry {
 public:
  using RegistrationFunction = std::function<void(TypeIDNameRegistry*)>;

  TypeIDNameRegistry();
  ~TypeIDNameRegistry() = default;

  TypeIDNameRegistry(const TypeIDNameRegistry&) = delete;
  TypeIDNameRegistry& operator=(const TypeIDNameRegistry&) = delete;

  template <typename T>
  void Register(llvm::StringRef type_name) {
    auto type_id = TypeID::get<T>();
    auto inserted = type_id_name_map_.try_emplace(type_id, type_name);
    assert(inserted.second && "duplicate typeid name registration");
    (void)inserted;
  }
  llvm::StringRef FindTypeIDSymbolName(TypeID type_id);

  void ForEach(std::function<void(llvm::StringRef, TypeID)> f) const {
    for (auto& kv : type_id_name_map_) f(kv.second, kv.first);
  }

 private:
  llvm::DenseMap<TypeID, llvm::StringRef> type_id_name_map_;
};

void RegisterStaticTypeIDName(TypeIDNameRegistry* typeid_name_registry);

void AddStaticTypeIDNameRegistration(
    TypeIDNameRegistry::RegistrationFunction registration);

#define XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(T, NAME) \
  XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION_IMPL(T, NAME, __COUNTER__)
#define XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION_IMPL(T, NAME, N) \
  XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION_IMPL_EXPAND(T, NAME, N)
#define XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION_IMPL_EXPAND(T, NAME, N) \
  static bool XLA_RUNTIME_static_typeid_name_##N##_registered = []() {      \
    auto func = [](::xla::runtime::TypeIDNameRegistry* registry) {          \
      registry->Register<::xla::runtime::Tagged<T>>(NAME);                  \
    };                                                                      \
    ::xla::runtime::AddStaticTypeIDNameRegistration(func);                  \
    return true;                                                            \
  }()

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_TYPE_ID_H_
