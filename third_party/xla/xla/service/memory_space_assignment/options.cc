/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/memory_space_assignment/options.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace xla {
namespace memory_space_assignment {

std::string PostAllocationTransformationUpdate::ToString() const {
  return absl::StrCat("to_be_removed: ",
                      absl::StrJoin(to_be_removed, ", ",
                                    [](std::string* out, const auto& entry) {
                                      absl::StrAppend(out, entry->name());
                                    }),
                      "\n", "update_use_map: ",
                      absl::StrJoin(update_use_map, ", ",
                                    [](std::string* out, const auto& entry) {
                                      absl::StrAppend(
                                          out, "<", entry.first.ToString(),
                                          " -> ", entry.second.ToString(), ">");
                                    }),
                      "\n");
}

}  // namespace memory_space_assignment
}  // namespace xla
