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

#include "xla/codegen/tiling/tiled_hlo_computation.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/name_uniquer.h"
#include "xla/util.h"

namespace xla {

namespace {
std::string InstructionsToString(
    absl::Span<const std::unique_ptr<TiledHloInstruction>> instructions,
    int64_t indent, NameUniquer& name_uniquer,
    absl::flat_hash_map<const TiledHloInstruction*, std::string>& tile_names) {
  std::string indentation(indent, ' ');
  std::stringstream ss;
  for (const std::unique_ptr<xla::TiledHloInstruction>& tiled_hlo :
       instructions) {
    absl::InlinedVector<std::string, 4> regions;
    for (const auto& region : tiled_hlo->hlo_regions()) {
      regions.push_back(
          InstructionsToString(region, indent + 4, name_uniquer, tile_names));
    }
    std::string tile_name = name_uniquer.GetUniqueName(
        absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0"));
    tile_names[tiled_hlo.get()] = tile_name;
    absl::InlinedVector<std::string, 4> operand_names;
    for (const auto& operand : tiled_hlo->operands()) {
      if (tile_names.contains(operand)) {
        operand_names.push_back(tile_names.at(operand));
      } else {
        LOG(ERROR) << "Operand " << operand->ToString()
                   << " not found in tile_names";
        operand_names.push_back(
            absl::StrCat("<error ", operand->hlo()->name(), ">"));
      }
    }
    ss << indentation << tile_name << " = "
       << HloOpcodeString(tiled_hlo->hlo()->opcode()) << "("
       << absl::StrJoin(operand_names, ", ") << ")\n";
    ss << tiled_hlo->ToString(indent + 2) << "\n";
    for (int i = 0; i < regions.size(); ++i) {
      ss << indentation << "  region." << i << " {\n" << regions[i] << "  }";
    }
  }
  return ss.str();
}
}  // namespace

std::string TiledHloComputation::ToString() const {
  NameUniquer name_uniquer("_");
  absl::flat_hash_map<const TiledHloInstruction*, std::string> tile_names;
  return InstructionsToString(instructions_, 0, name_uniquer, tile_names);
}

}  // namespace xla
