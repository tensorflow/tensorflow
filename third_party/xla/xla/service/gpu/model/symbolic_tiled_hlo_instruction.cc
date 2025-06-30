/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"

#include <sstream>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/service/gpu/model/symbolic_tile.h"

namespace xla {
namespace gpu {

std::string SymbolicTiledHloInstruction::ToString(
    absl::string_view field_separator) const {
  std::stringstream ss;
  ss << "hlo: " << hlo_->ToString() << field_separator;
  if (symbolic_tile_.has_value()) {
    ss << symbolic_tile().ToString();
  } else {
    ss << "(no symbolic tile)";
  }
  ss << field_separator;
  ss << "indexing map: " << indexing_map_;
  if (!runtime_variables_.empty()) {
    ss << field_separator;
    ss << "runtime operands: (";
    for (const auto& rt_var : runtime_variables_) {
      ss << rt_var->ToString() << field_separator;
    }
    ss << ")";
  }
  return ss.str();
}

}  // namespace gpu
}  // namespace xla
