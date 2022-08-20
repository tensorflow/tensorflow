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
#include "tensorflow/compiler/xla/runtime/type_id.h"

#include <vector>

namespace xla {
namespace runtime {

TypeIDNameRegistry::TypeIDNameRegistry() : type_id_name_map_() {}

llvm::StringRef TypeIDNameRegistry::FindTypeIDSymbolName(TypeID type_id) {
  auto it = type_id_name_map_.find(type_id);
  if (it == type_id_name_map_.end()) return "";
  return it->second;
}

static std::vector<TypeIDNameRegistry::RegistrationFunction>*
GetTypeIDNameRegistrations() {
  static auto* ret = new std::vector<TypeIDNameRegistry::RegistrationFunction>;
  return ret;
}

void RegisterStaticTypeIDName(TypeIDNameRegistry* typeid_name_registry) {
  for (const auto& func : *GetTypeIDNameRegistrations())
    func(typeid_name_registry);
}

void AddStaticTypeIDNameRegistration(
    TypeIDNameRegistry::RegistrationFunction registration) {
  GetTypeIDNameRegistrations()->push_back(registration);
}
}  // namespace runtime
}  // namespace xla
