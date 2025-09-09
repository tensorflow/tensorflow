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

#include "xla/hlo/translate/mhlo_to_hlo/location_exporter.h"

#include <memory>
#include <optional>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/attributes.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/hlo/translate/mhlo_to_hlo/stack_frame_index_builder.h"
#include "xla/xla_data.pb.h"

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
      if (name.starts_with(kOriginalValueAttr)) {
        continue;
      }
      if (name.ends_with(":")) {
        locs.push_back(name_loc.getChildLoc());
      } else {
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
      if (op_type.starts_with(kOriginalValueAttr)) {
        continue;
      }
      if (op_type.ends_with(":")) {
        op_type = op_type.substr(0, op_type.size() - 1);
        loc_op_types.push_back(op_type);
      } else {
        locs.push_back(name_loc.getChildLoc());
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

static std::shared_ptr<xla::OriginalValue> GetOriginalValueFromLoc(
    Location loc) {
  llvm::StringRef loc_original_value;
  llvm::SmallVector<Location, 8> locs;
  locs.push_back(loc);

  while (!locs.empty()) {
    Location curr_loc = locs.pop_back_val();

    if (auto name_loc = mlir::dyn_cast<NameLoc>(curr_loc)) {
      auto original_value = name_loc.getName().strref().split('@').first;
      if (!original_value.starts_with(kOriginalValueAttr)) {
        continue;
      }
      loc_original_value = original_value.split('=').second;
      break;
    }
    if (auto fused_loc = mlir::dyn_cast<FusedLoc>(curr_loc)) {
      // Push all locations in FusedLoc in reverse order, so locations are
      // visited based on order in FusedLoc.
      auto reversed_fused_locs = llvm::reverse(fused_loc.getLocations());
      locs.append(reversed_fused_locs.begin(), reversed_fused_locs.end());
    }
  }

  auto original_value =
      xla::ParseOriginalValue(xla::ToStringView(loc_original_value));
  if (!original_value.ok()) {
    return nullptr;
  }
  return original_value.value();
}

static void SetSourceFileAndLine(Location loc, xla::OpMetadata& metadata) {
  if (auto file_line_col_loc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    metadata.set_source_file(file_line_col_loc.getFilename().str());
    metadata.set_source_line(file_line_col_loc.getLine());
    metadata.set_source_end_line(file_line_col_loc.getEndLine());
    metadata.set_source_column(file_line_col_loc.getColumn());
    metadata.set_source_end_column(file_line_col_loc.getEndColumn());
  } else if (auto fused_loc = mlir::dyn_cast<FusedLoc>(loc)) {
    for (Location it : fused_loc.getLocations()) {
      SetSourceFileAndLine(it, metadata);
    }
  }
}

static bool IsFrameNameLocation(mlir::Location location) {
  return isa<mlir::NameLoc>(location) &&
         isa<mlir::FileLineColLoc>(cast<mlir::NameLoc>(location).getChildLoc());
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

  // Skip all leading names that are not frame names, e.g., op name and op type
  // attributes found above.
  while (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
    if (IsFrameNameLocation(name_loc)) {
      break;
    }
    loc = name_loc.getChildLoc();
  }

  if (isa<mlir::UnknownLoc>(loc)) {
    return metadata;
  }

  if (frame_index_builder != nullptr) {
    auto result = frame_index_builder->AddCallStackAndGetFirstFrameId(loc);
    if (result.last_frame_id != mlir::StackFrameIndexBuilder::kInvalidIndex) {
      metadata.set_stack_frame_id(result.last_frame_id);
      // TODO(b/311155137): Remove when profiler will support stack traces.
      metadata.set_source_file(result.last_frame_file);
      metadata.set_source_line(result.last_frame_line);
      metadata.set_source_end_line(result.last_frame_end_line);
      metadata.set_source_column(result.last_frame_column);
      metadata.set_source_end_column(result.last_frame_end_column);
      return metadata;
    }
  }

  SetSourceFileAndLine(loc, metadata);
  return metadata;
}

std::string GetDebugNameFromLocation(mlir::Location loc) {
  return GetNameFromLocImpl(loc);
}

std::optional<xla::OriginalValueProto> CreateOriginalValueFromOp(
    mlir::Operation* op) {
  mlir::Location loc = op->getLoc();
  return CreateOriginalValueFromLocation(loc);
}

std::optional<xla::OriginalValueProto> CreateOriginalValueFromLocation(
    mlir::Location loc) {
  if (isa<mlir::UnknownLoc>(loc)) {
    return std::nullopt;
  }

  if (std::shared_ptr<xla::OriginalValue> original_value =
          GetOriginalValueFromLoc(loc)) {
    return original_value->ToProto();
  }

  return std::nullopt;
}

}  // namespace mhlo
}  // namespace mlir
