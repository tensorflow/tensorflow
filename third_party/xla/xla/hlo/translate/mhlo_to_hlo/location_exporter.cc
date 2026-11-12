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

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/translate/mhlo_to_hlo/stack_frame_index_builder.h"
#include "xla/xla_data.pb.h"

namespace mlir {
namespace mhlo {
namespace {

// MLIR uniques locations, so nested locations can share sub-locations and form
// a DAG whose expansion into a tree is exponentially larger than the module
// that encodes it. The traversals below expand the location they are given, so
// each of them visits at most this many (not necessarily distinct) locations
// and ignores the rest. Locations emitted by frontends are far smaller.
constexpr int kMaxVisitedLocations = 4096;

// Maximum size of an exported op name, in bytes. Names are not uniqued when
// they are exported, so a long name shared by many sub-locations would
// otherwise be repeated for each of them.
constexpr size_t kMaxOpNameSize = 4096;

bool IsFrameNameLocation(mlir::Location location);

// Extracts the op name(s) from `root_loc`. The names in a `FusedLoc` are
// separated by `;`, and the caller names of a `CallSiteLoc` are separated from
// its callee names by `/`. On the caller side of a `CallSiteLoc`, only explicit
// op names from inlined call sites are extracted, ignoring source stack frames.
// The result is cut before the first name that would make it exceed
// `kMaxOpNameSize`.
std::string GetNameFromLocImpl(Location root_loc) {
  // A location whose sub-locations are being visited.
  struct Item {
    Location loc;
    // Whether `loc` is on the caller side of a `CallSiteLoc`.
    bool is_caller;
    // Number of sub-locations of `loc` visited so far.
    size_t num_visited;
    // Size of `names` before the sub-locations of `loc` were visited.
    size_t names_size;
  };
  llvm::SmallVector<Item, 8> worklist;
  int budget = kMaxVisitedLocations;
  std::string names;
  // Separator to insert before the next name.
  char separator = ';';
  bool is_full = false;

  // Appends `name` to `names`, unless it is empty or does not fit.
  auto append_name = [&](llvm::StringRef name) {
    if (name.empty()) {
      return;
    }
    const size_t separator_size = names.empty() ? 0 : 1;
    if (names.size() + separator_size + name.size() > kMaxOpNameSize) {
      is_full = true;
      return;
    }
    if (!names.empty()) {
      names.push_back(separator);
    }
    names.append(name.data(), name.size());
    separator = ';';
  };

  // Appends the name in `loc`, or queues `loc` to visit its sub-locations.
  auto visit = [&](Location loc, bool is_caller) {
    --budget;
    if (auto name_loc = mlir::dyn_cast<NameLoc>(loc)) {
      if (is_caller && IsFrameNameLocation(name_loc)) {
        return;
      }
      // Add name in NameLoc. For NameLoc we also account for names due to ops
      // in functions where the op's name is first.
      llvm::StringRef name = name_loc.getName().strref().split('@').first;
      // If the name is for the op type, the op name is in the child location.
      if (!name.ends_with(":")) {
        append_name(name);
        return;
      }
    }
    worklist.push_back(Item{loc, is_caller, 0, names.size()});
  };

  visit(root_loc, /*is_caller=*/false);
  while (!worklist.empty() && budget > 0 && !is_full) {
    Item& item = worklist.back();
    std::optional<Location> sub_loc;
    bool is_caller = item.is_caller;
    if (auto call_loc = mlir::dyn_cast<CallSiteLoc>(item.loc)) {
      // Combine caller and callee op names (ignoring bare caller stack frames).
      const bool has_names = names.size() > item.names_size;
      if (item.num_visited == 0) {
        sub_loc = call_loc.getCaller();
        is_caller = true;
      } else if (item.num_visited == 1) {
        sub_loc = call_loc.getCallee();
        // The first callee name follows the caller names.
        if (has_names) {
          separator = '/';
        }
      } else if (has_names) {
        // The next name does not belong to this `CallSiteLoc`.
        separator = ';';
      }
    } else if (auto name_loc = mlir::dyn_cast<NameLoc>(item.loc)) {
      if (item.num_visited == 0) {
        sub_loc = name_loc.getChildLoc();
      }
    } else if (auto fused_loc = mlir::dyn_cast<FusedLoc>(item.loc)) {
      llvm::ArrayRef<Location> locs = fused_loc.getLocations();
      if (item.num_visited < locs.size()) {
        sub_loc = locs[item.num_visited];
      }
    }
    if (!sub_loc) {
      worklist.pop_back();
      continue;
    }
    ++item.num_visited;
    visit(*sub_loc, is_caller);
  }

  return names;
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
      } else {
        locs.push_back(name_loc.getChildLoc());
      }
    } else if (auto call_loc = mlir::dyn_cast<CallSiteLoc>(curr_loc)) {
      // Use location of the Callee to generate the name.
      locs.push_back(call_loc.getCallee());
    } else if (auto fused_loc = mlir::dyn_cast<FusedLoc>(curr_loc)) {
      // The first location is reserved for op_type.
      if (!fused_loc.getLocations().empty()) {
        locs.push_back(fused_loc.getLocations()[0]);
      }
    }
  }

  return llvm::join(loc_op_types.begin(), loc_op_types.end(), ";");
}

static void SetSourceFileAndLine(Location root_loc, xla::OpMetadata& metadata) {
  // `FusedLoc`s whose sub-locations are being visited, with the number of
  // sub-locations visited so far.
  llvm::SmallVector<std::pair<FusedLoc, size_t>, 4> worklist;
  int budget = kMaxVisitedLocations;

  auto visit = [&](Location loc) {
    --budget;
    if (auto file_line_col_loc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
      metadata.set_source_file(file_line_col_loc.getFilename().str());
      metadata.set_source_line(file_line_col_loc.getLine());
      metadata.set_source_end_line(file_line_col_loc.getEndLine());
      metadata.set_source_column(file_line_col_loc.getColumn());
      metadata.set_source_end_column(file_line_col_loc.getEndColumn());
    } else if (auto fused_loc = mlir::dyn_cast<FusedLoc>(loc)) {
      worklist.emplace_back(fused_loc, 0);
    }
  };

  visit(root_loc);
  while (!worklist.empty() && budget > 0) {
    auto& [fused_loc, num_visited] = worklist.back();
    llvm::ArrayRef<Location> locs = fused_loc.getLocations();
    if (num_visited == locs.size()) {
      worklist.pop_back();
    } else {
      visit(locs[num_visited++]);
    }
  }
}

bool IsFrameNameLocation(mlir::Location location) {
  return isa<mlir::NameLoc>(location) &&
         isa<mlir::FileLineColLoc>(cast<mlir::NameLoc>(location).getChildLoc());
}
}  // namespace

xla::OpMetadata CreateOpMetadataFromLocation(
    mlir::Operation* op, mlir::StackFrameIndexBuilder* frame_index_builder) {
  xla::OpMetadata metadata;
  mlir::Location loc = op->getLoc();
  if (isa<mlir::UnknownLoc>(loc)) {
    return metadata;
  }

  std::string name = GetNameFromLocImpl(loc);
  metadata.set_op_name(name);
  std::string op_type = GetOpTypeFromLoc(loc);
  metadata.set_op_type(op_type);

  // TODO(b/503147499): Consider plumbing metadata payload directly from the
  // frontend.
  if (auto attrs =
          op->getAttrOfType<mlir::DictionaryAttr>("mhlo.frontend_attributes")) {
    if (auto payload = mlir::dyn_cast_or_null<mlir::StringAttr>(
            attrs.get("xla_metadata_payload"))) {
      metadata.mutable_metadata_payload()->set_value(payload.getValue().str());
    }
  }

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
      return metadata;
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
