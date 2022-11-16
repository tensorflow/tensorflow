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

#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/location_exporter.h"

#include <string>

#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/Builders.h"  // from @llvm-project

namespace mlir {
namespace mhlo {

static std::string GetNameFromLocImpl(Location loc) {
  llvm::SmallVector<llvm::StringRef, 8> loc_names;
  llvm::SmallVector<Location, 8> locs;
  locs.push_back(loc);

  while (!locs.empty()) {
    Location curr_loc = locs.pop_back_val();

    if (auto name_loc = curr_loc.dyn_cast<NameLoc>()) {
      // Add name in NameLoc. For NameLoc we also account for names due to ops
      // in functions where the op's name is first.
      auto name = name_loc.getName().strref().split('@').first;
      // Skip if the name is for op type.
      if (!name.endswith(":")) {
        loc_names.push_back(name);
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

  return llvm::join(loc_names.begin(), loc_names.end(), ";");
}

static std::string GetOpTypeFromLoc(Location loc) {
  llvm::SmallVector<llvm::StringRef, 1> loc_op_types;
  llvm::SmallVector<Location, 8> locs;
  locs.push_back(loc);

  while (!locs.empty()) {
    Location curr_loc = locs.pop_back_val();

    if (auto name_loc = curr_loc.dyn_cast<NameLoc>()) {
      // Add name in NameLoc. For NameLoc we also account for names due to ops
      // in functions where the op's name is first.
      auto op_type = name_loc.getName().strref().split('@').first;
      if (op_type.endswith(":")) {
        op_type = op_type.substr(0, op_type.size() - 1);
        loc_op_types.push_back(op_type);
      }
      continue;
    } else if (auto call_loc = curr_loc.dyn_cast<CallSiteLoc>()) {
      // Use location of the Callee to generate the name.
      locs.push_back(call_loc.getCallee());
      continue;
    } else if (auto fused_loc = curr_loc.dyn_cast<FusedLoc>()) {
      // The first location is reserved for op_type.
      if (!fused_loc.getLocations().empty())
        locs.push_back(fused_loc.getLocations()[0]);
      continue;
    }

    // Location is not a supported, so an empty StringRef is added.
    loc_op_types.push_back(llvm::StringRef());
  }

  return llvm::join(loc_op_types.begin(), loc_op_types.end(), ";");
}

xla::OpMetadata CreateOpMetadataFromLocation(mlir::Operation* op) {
  xla::OpMetadata metadata;
  mlir::Location loc = op->getLoc();
  if (loc.isa<mlir::UnknownLoc>()) return metadata;

  std::string name = GetNameFromLocImpl(loc);
  metadata.set_op_name(name);
  std::string op_type = GetOpTypeFromLoc(loc);
  metadata.set_op_type(op_type);

  if (auto name_loc = op->getLoc().dyn_cast<mlir::NameLoc>()) {
    loc = name_loc.getChildLoc();
    if (loc.isa<mlir::UnknownLoc>()) return metadata;
  }

  if (auto file_line_col_loc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    metadata.set_source_file(file_line_col_loc.getFilename().str());
    metadata.set_source_line(file_line_col_loc.getLine());
  }

  return metadata;
}

std::string GetDebugNameFromLocation(mlir::Location loc) {
  return GetNameFromLocImpl(loc);
}

}  // namespace mhlo
}  // namespace mlir
