/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir

static inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return absl::string_view(ref.data(), ref.size());
}

static inline llvm::StringRef StringViewToRef(absl::string_view view) {
  return llvm::StringRef(view.data(), view.size());
}

namespace tensorflow {

OpOrArgNameMapper::~OpOrArgNameMapper() {}

llvm::StringRef OpOrArgNameMapper::GetUniqueName(llvm::StringRef prefix) {
  // Insert/find if prefix is unique.
  auto prefix_it = name_to_count_.try_emplace(prefix, 0);
  if (prefix_it.second && IsUnique(prefix)) {
    // Name is unique, increment count and return string name backed by
    // `name_to_count_`.
    ++prefix_it.first->second;
    return prefix_it.first->first();
  }

  // Add increasing number (count) to end of prefix until it is determined
  // to be unique.
  auto& val = prefix_it.first->second;
  llvm::SmallString<64> probe_name(prefix);
  while (true) {
    probe_name.resize(prefix.size());
    // TODO(jpienaar): Subtract one so that the initial suffix is 0 instead
    // of 1.
    // TODO(jpienaar): Switch to radix 36 and update tests.
    llvm::APInt(32, val++).toString(probe_name, /*Radix=*/10, /*Signed=*/false);
    if (IsUnique(probe_name)) {
      // Insert/find if prefix with appended number is unique.
      auto probe_name_it = name_to_count_.try_emplace(probe_name, 1);
      if (probe_name_it.second) {
        // Name is unique, return string name backed by `name_to_count_`.
        return probe_name_it.first->first();
      }
    }
  }
}

llvm::StringRef OpOrArgNameMapper::GetUniqueName(OpOrArg op_or_arg) {
  auto& name = op_or_arg_to_name_[op_or_arg];
  if (!name.empty()) return StringViewToRef(name);
  // Update the value in the map with unique name.
  llvm::StringRef ref = GetUniqueName(GetName(op_or_arg));
  name = StringRefToView(ref);
  return ref;
}

absl::string_view OpOrArgNameMapper::GetUniqueNameView(OpOrArg op_or_arg) {
  auto& name = op_or_arg_to_name_[op_or_arg];
  if (!name.empty()) return name;
  // Update the value in the map with unique name.
  name = StringRefToView(GetUniqueName(GetName(op_or_arg)));
  return name;
}

int OpOrArgNameMapper::InitOpName(OpOrArg op_or_arg, llvm::StringRef name) {
  auto it = name_to_count_.try_emplace(name, 0);
  op_or_arg_to_name_[op_or_arg] = StringRefToView(it.first->first());
  return it.first->second++;
}

bool OpOrArgNameMapper::IsUnique(llvm::StringRef name) { return true; }

namespace {
// Derives name from location.
std::string GetNameFromLoc(mlir::Location loc) {
  llvm::SmallVector<llvm::StringRef, 8> loc_names;
  llvm::SmallVector<mlir::Location, 8> locs;
  locs.push_back(loc);
  bool names_is_nonempty = false;

  while (!locs.empty()) {
    mlir::Location curr_loc = locs.pop_back_val();

    if (auto name_loc = curr_loc.dyn_cast<mlir::NameLoc>()) {
      // Add name in NameLoc.
      loc_names.push_back(name_loc.getName().strref());
      if (!name_loc.getName().strref().empty()) names_is_nonempty = true;
      continue;
    } else if (auto call_loc = curr_loc.dyn_cast<mlir::CallSiteLoc>()) {
      // Add name if CallSiteLoc's callee has a NameLoc (as should be the
      // case if imported with DebugInfo).
      if (auto name_loc = call_loc.getCallee().dyn_cast<mlir::NameLoc>()) {
        loc_names.push_back(name_loc.getName().strref());
        if (!name_loc.getName().strref().empty()) names_is_nonempty = true;
        continue;
      }
    } else if (auto fused_loc = curr_loc.dyn_cast<mlir::FusedLoc>()) {
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
}  // anonymous namespace

std::string OpOrArgLocNameMapper::GetName(OpOrArg op_or_arg) {
  if (auto* op = op_or_arg.dyn_cast<mlir::Operation*>()) {
    auto name_from_loc = GetNameFromLoc(op->getLoc());
    if (!name_from_loc.empty()) return name_from_loc;
    // If the location is none of the expected types, then simply use name
    // generated using the op type.
    return op->getName().getStringRef();
  }

  if (auto* arg = op_or_arg.dyn_cast<mlir::BlockArgument*>())
    return GetNameFromLoc(arg->getLoc());

  return "";
}

std::string OpOrArgStripNameMapper::GetName(OpOrArg op_or_arg) {
  return llvm::APInt(32, count_++).toString(/*Radix=*/36, /*Signed=*/false);
}

}  // namespace tensorflow
