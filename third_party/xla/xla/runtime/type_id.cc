/* Copyright 2022 The OpenXLA Authors.

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
#include "xla/runtime/type_id.h"

#include <string_view>
#include <vector>

namespace xla {
namespace runtime {

std::string_view TypeIDNameRegistry::FindTypeIDSymbolName(TypeID type_id) {
  auto it = type_id_name_map_.find(type_id);
  if (it == type_id_name_map_.end()) return "";
  return it->second;
}

}  // namespace runtime
}  // namespace xla
