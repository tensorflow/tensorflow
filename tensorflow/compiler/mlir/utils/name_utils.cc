/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/utils/name_utils.h"

#include <cctype>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project

namespace mlir {

namespace {
// Checks if a character is legal for a TensorFlow node name, with special
// handling if a character is at the beginning.
bool IsLegalChar(char c, bool first_char) {
  if (isalpha(c)) return true;
  if (isdigit(c)) return true;
  if (c == '.') return true;
  if (c == '_') return true;

  // First character of a node name can only be a letter, digit, dot or
  // underscore.
  if (first_char) return false;

  if (c == '/') return true;
  if (c == '-') return true;

  return false;
}
}  // anonymous namespace

void LegalizeNodeName(std::string& name) {
  if (name.empty()) return;

  if (!IsLegalChar(name[0], /*first_char=*/true)) name[0] = '.';

  for (char& c : llvm::drop_begin(name, 1))
    if (!IsLegalChar(c, /*first_char=*/false)) c = '.';
}

std::string GetNameFromLoc(Location loc) {
  llvm::SmallVector<llvm::StringRef, 8> loc_names;
  llvm::SmallVector<Location, 8> locs;
  locs.push_back(loc);
  bool names_is_nonempty = false;

  while (!locs.empty()) {
    Location curr_loc = locs.pop_back_val();

    if (auto name_loc = curr_loc.dyn_cast<NameLoc>()) {
      // Add name in NameLoc. For NameLoc we also account for names due to ops
      // in functions where the op's name is first.
      auto name = name_loc.getName().strref().split('@').first;
      // Skip if the name is for op type.
      if (!name.endswith(":")) {
        loc_names.push_back(name);
        if (!name.empty()) names_is_nonempty = true;
      }
      continue;
    } else if (auto call_loc = curr_loc.dyn_cast<CallSiteLoc>()) {
      // Use location of the Callee to generate the name.
      locs.push_back(call_loc.getCallee());
      continue;
    } else if (auto fused_loc = curr_loc.dyn_cast<FusedLoc>()) {
      // Push all locations in FusedLoc in reverse order, so locations are
      // visited based on order in FusedLoc.
      auto reversed_fused_locs = llvm::reverse(fused_loc.getLocations());
      locs.append(reversed_fused_locs.begin(), reversed_fused_locs.end());
      continue;
    }

    // Location is not a supported, so an empty StringRef is added.
    loc_names.push_back(llvm::StringRef());
  }

  if (names_is_nonempty)
    return llvm::join(loc_names.begin(), loc_names.end(), ";");

  return "";
}

}  // namespace mlir
