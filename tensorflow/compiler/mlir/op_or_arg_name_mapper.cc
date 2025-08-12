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

#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/utils/name_utils.h"

static inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return absl::string_view(ref.data(), ref.size());
}

static inline llvm::StringRef StringViewToRef(absl::string_view view) {
  return llvm::StringRef(view.data(), view.size());
}

namespace tensorflow {

OpOrArgNameMapper::~OpOrArgNameMapper() = default;

llvm::StringRef OpOrArgNameMapper::GetUniqueName(llvm::StringRef prefix,
                                                 int hash_value) {
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
  auto prefix_name = hash_value == 0 ? prefix.str() + GetSuffixSeparator().str()
                                     : prefix.str() + GetDashSeparator().str() +
                                           std::to_string(hash_value) +
                                           GetDashSeparator().str();
  llvm::SmallString<64> probe_name(prefix_name);
  const int probe_prefix_size = probe_name.size();
  while (true) {
    probe_name.resize(probe_prefix_size);
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

std::optional<llvm::StringRef> OpOrArgNameMapper::GetMappedName(
    OpOrVal op_or_val) {
  auto name = GetMappedNameView(op_or_val);
  if (name.has_value()) return StringViewToRef(name.value());
  return std::nullopt;
}

std::optional<absl::string_view> OpOrArgNameMapper::GetMappedNameView(
    OpOrVal op_or_val) {
  auto& name = op_or_val_to_name_[op_or_val];
  if (!name.empty()) return name;
  return std::nullopt;
}

llvm::StringRef OpOrArgNameMapper::GetUniqueName(OpOrVal op_or_val,
                                                 int hash_value) {
  auto& name = op_or_val_to_name_[op_or_val];
  if (!name.empty()) return StringViewToRef(name);
  // Update the value in the map with unique name.
  llvm::StringRef ref = GetUniqueName(GetName(op_or_val), hash_value);
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

std::string OpOrArgLocNameMapper::GetName(OpOrVal op_or_val) {
  if (auto* op = op_or_val.dyn_cast<mlir::Operation*>()) {
    auto name_from_loc = mlir::GetNameFromLoc(op->getLoc());
    if (!name_from_loc.empty()) return name_from_loc;
    // If the location is none of the expected types, then simply use name
    // generated using the op type.
    return std::string(op->getName().getStringRef());
  }
  auto val = op_or_val.dyn_cast<mlir::Value>();
  auto name_from_loc = mlir::GetNameFromLoc(val.getLoc());
  if (!name_from_loc.empty()) return name_from_loc;
  // If the location is none of the expected types, then simply use name
  // generated using the op type. Follow TF convention and append the result
  // index unless 0.
  if (auto result = mlir::dyn_cast<mlir::OpResult>(val)) {
    if (result.getResultNumber() > 0)
      return llvm::formatv("{0}:{1}",
                           result.getOwner()->getName().getStringRef(),
                           result.getResultNumber());
    return std::string(result.getOwner()->getName().getStringRef());
  }
  // Use the ASM syntax for BlockArgument
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    return "arg" + std::to_string(arg.getArgNumber());
  }
  return "";
}

std::string OpOrArgStripNameMapper::GetName(OpOrVal op_or_val) {
  return llvm::toString(llvm::APInt(32, count_++),
                        /*Radix=*/36, /*Signed=*/false);
}

}  // namespace tensorflow
