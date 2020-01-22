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
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project

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

llvm::StringRef OpOrArgNameMapper::GetUniqueName(OpOrVal op_or_val) {
  auto& name = op_or_val_to_name_[op_or_val];
  if (!name.empty()) return StringViewToRef(name);
  // Update the value in the map with unique name.
  llvm::StringRef ref = GetUniqueName(GetName(op_or_val));
  name = StringRefToView(ref);
  return ref;
}

absl::string_view OpOrArgNameMapper::GetUniqueNameView(OpOrVal op_or_val) {
  auto& name = op_or_val_to_name_[op_or_val];
  if (!name.empty()) return name;
  // Update the value in the map with unique name.
  name = StringRefToView(GetUniqueName(GetName(op_or_val)));
  return name;
}

int OpOrArgNameMapper::InitOpName(OpOrVal op_or_val, llvm::StringRef name) {
  auto it = name_to_count_.try_emplace(name, 0);
  auto inserted = op_or_val_to_name_.try_emplace(
      op_or_val, StringRefToView(it.first->first()));
  (void)inserted;
  // TODO(jpienaar): Debug cases where we expect this behavior.
  // assert(inserted.second && "op_or_val already initialized");
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
      // Add name in NameLoc. For NameLoc we also account for names due to ops
      // in functions where the op's name is first.
      auto name = name_loc.getName().strref().split('@').first;
      loc_names.push_back(name);
      if (!name.empty()) names_is_nonempty = true;
      continue;
    } else if (auto call_loc = curr_loc.dyn_cast<mlir::CallSiteLoc>()) {
      // Add name if CallSiteLoc's callee has a NameLoc (as should be the
      // case if imported with DebugInfo).
      if (auto name_loc = call_loc.getCallee().dyn_cast<mlir::NameLoc>()) {
        auto name = name_loc.getName().strref().split('@').first;
        loc_names.push_back(name);
        if (!name.empty()) names_is_nonempty = true;
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

std::string OpOrArgLocNameMapper::GetName(OpOrVal op_or_val) {
  if (auto* op = op_or_val.dyn_cast<mlir::Operation*>()) {
    auto name_from_loc = GetNameFromLoc(op->getLoc());
    if (!name_from_loc.empty()) return name_from_loc;
    // If the location is none of the expected types, then simply use name
    // generated using the op type.
    return op->getName().getStringRef();
  }
  auto val = op_or_val.dyn_cast<mlir::Value>();
  auto name_from_loc = GetNameFromLoc(val.getLoc());
  if (!name_from_loc.empty()) return name_from_loc;
  // If the location is none of the expected types, then simply use name
  // generated using the op type. Follow TF convention and append the result
  // index unless 0.
  if (auto result = val.dyn_cast<mlir::OpResult>()) {
    if (result.getResultNumber() > 0)
      return llvm::formatv("{0}:{1}",
                           result.getOwner()->getName().getStringRef(),
                           result.getResultNumber());
    return result.getOwner()->getName().getStringRef();
  }
  return "";
}

std::string OpOrArgStripNameMapper::GetName(OpOrVal op_or_val) {
  return llvm::APInt(32, count_++).toString(/*Radix=*/36, /*Signed=*/false);
}

}  // namespace tensorflow
