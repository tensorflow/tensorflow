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

#include "xla/translate/mhlo_to_hlo/location_exporter.h"

#include <string>

#include "absl/log/log.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/translate/mhlo_to_hlo/stack_frame_index_builder.h"

namespace mlir {
namespace mhlo {

static std::string GetNameFromLocImpl(Location loc) {
  llvm::SmallVector<llvm::StringRef, 8> loc_names;
  llvm::SmallVector<Location, 8> locs;
  locs.push_back(loc);

  while (!locs.empty()) {
    Location curr_loc = locs.pop_back_val();

    if (auto name_loc = mlir::dyn_cast<NameLoc>(curr_loc)) {
      // Add name in NameLoc. For NameLoc we also account for names due to ops
      // in functions where the op's name is first.
      auto name = name_loc.getName().strref().split('@').first;
      // Skip if the name is for op type.
      if (!name.ends_with(":")) {
        loc_names.push_back(name);
      }
    } else if (auto call_loc = mlir::dyn_cast<CallSiteLoc>(curr_loc)) {
      // Use location of the Callee to generate the name.
      locs.push_back(call_loc.getCallee());
    } else if (auto fused_loc = mlir::dyn_cast<FusedLoc>(curr_loc)) {
      // Push all locations in FusedLoc in reverse order, so locations are
      // visited based on order in FusedLoc.
      auto reversed_fused_locs = llvm::reverse(fused_loc.getLocations());
      locs.append(reversed_fused_locs.begin(), reversed_fused_locs.end());
    }
  }

  return llvm::join(loc_names.begin(), loc_names.end(), ";");
}

static std::string GetOpTypeFromLoc(Location loc) {
  llvm::SmallVector<llvm::StringRef, 1> loc_op_types;
  llvm::SmallVector<Location, 8> locs;
  locs.push_back(loc);

  while (!locs.empty()) {
    Location curr_loc = locs.pop_back_val();

    if (auto name_loc = mlir::dyn_cast<NameLoc>(curr_loc)) {
      // Add name in NameLoc. For NameLoc we also account for names due to ops
      // in functions where the op's name is first.
      auto op_type = name_loc.getName().strref().split('@').first;
      if (op_type.ends_with(":")) {
        op_type = op_type.substr(0, op_type.size() - 1);
        loc_op_types.push_back(op_type);
      }
    } else if (auto call_loc = mlir::dyn_cast<CallSiteLoc>(curr_loc)) {
      // Use location of the Callee to generate the name.
      locs.push_back(call_loc.getCallee());
    } else if (auto fused_loc = mlir::dyn_cast<FusedLoc>(curr_loc)) {
      // The first location is reserved for op_type.
      if (!fused_loc.getLocations().empty())
        locs.push_back(fused_loc.getLocations()[0]);
    }
  }

  return llvm::join(loc_op_types.begin(), loc_op_types.end(), ";");
}

static void SetSourceFileAndLine(Location loc, xla::OpMetadata& metadata) {
  if (auto file_line_col_loc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    metadata.set_source_file(file_line_col_loc.getFilename().str());
    metadata.set_source_line(file_line_col_loc.getLine());
  } else if (auto fused_loc = mlir::dyn_cast<FusedLoc>(loc)) {
    for (Location it : fused_loc.getLocations()) {
      SetSourceFileAndLine(it, metadata);
    }
  }
}

xla::OpMetadata CreateOpMetadataFromLocation(
    mlir::Operation* op, mlir::StackFrameIndexBuilder* frame_index_builder) {
  xla::OpMetadata metadata;
  mlir::Location loc = op->getLoc();
  if (isa<mlir::UnknownLoc>(loc)) return metadata;

  std::string name = GetNameFromLocImpl(loc);
  metadata.set_op_name(name);
  std::string op_type = GetOpTypeFromLoc(loc);
  metadata.set_op_type(op_type);

  if (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
    loc = name_loc.getChildLoc();
    if (isa<mlir::UnknownLoc>(loc)) return metadata;

    if (frame_index_builder != nullptr) {
      auto result = frame_index_builder->AddCallStackAndGetFirstFrameId(loc);
      metadata.set_stack_frame_id(result.last_frame_id);
      // TODO(b/311155137): Remove when profiler will support stack traces.
      metadata.set_source_file(result.last_frame_file);
      metadata.set_source_line(result.last_frame_line);
    }
  }

  SetSourceFileAndLine(loc, metadata);
  return metadata;
}

std::string GetDebugNameFromLocation(mlir::Location loc) {
  return GetNameFromLocImpl(loc);
}

}  // namespace mhlo
}  // namespace mlir
