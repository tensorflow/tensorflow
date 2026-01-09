/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/tiling/symbolic_tiled_region.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/codegen/tiling/symbolic_tiled_hlo_instruction.h"

namespace xla {

std::string SymbolicTiledRegion::ToString(
    absl::string_view field_separator) const {
  return absl::StrCat(
      "SymbolicTiledRegion", field_separator,
      SymbolicTiledHloInstruction::ToString(field_separator), field_separator,
      "region instructions: (", field_separator,
      absl::StrJoin(
          instructions_, field_separator,
          [&](std::string* out,
              const std::unique_ptr<SymbolicTiledHloInstruction>& instruction) {
            absl::StrAppend(out, instruction->ToString(field_separator));
          }),
      field_separator, ")");
}

}  // namespace xla
