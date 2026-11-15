/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/export/tensor_names.h"

#include <cstddef>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::TFL::fb_export {
namespace {

using llvm::StringRef;

absl::string_view ToStringView(StringRef s) {
  return absl::string_view(s.data(), s.size());
}

// Returns the name of `loc` without its JAX `@...` suffix.
StringRef GetBaseName(NameLoc loc) {
  return loc.getName().strref().split('@').first;
}

// Returns true for a Python traceback frame emitted by JAX: a NameLoc with a
// flat function name (no '/') around a FileLineColLoc.
bool IsFrameLoc(Location loc) {
  auto name_loc = dyn_cast<NameLoc>(loc);
  return name_loc && isa<FileLineColLoc>(name_loc.getChildLoc()) &&
         !name_loc.getName().strref().contains('/');
}

// Skips JAX op type wrappers, i.e. NameLocs whose name ends with ':' (e.g.
// "pjit:", "composite:").
Location SkipOpTypeWrappers(Location loc) {
  while (auto name_loc = dyn_cast<NameLoc>(loc)) {
    if (IsFrameLoc(name_loc) || !GetBaseName(name_loc).ends_with(":")) break;
    loc = name_loc.getChildLoc();
  }
  return loc;
}

// Joins two scopes with '/'. A trailing `jit(_where)` caller scope (JAX's
// internal wrapper of `jnp.where`) is dropped when the callee has a name of
// its own, so that distinct `jnp.where`s stay distinct leaf ops.
std::string JoinScopes(StringRef caller, StringRef callee) {
  constexpr StringRef kJitWhere = "jit(_where)";
  if (!callee.empty()) {
    if (caller == kJitWhere) {
      caller = "";
    } else if (caller.ends_with(absl::StrCat("/", ToStringView(kJitWhere)))) {
      caller = caller.drop_back(kJitWhere.size() + 1);
    }
  }
  if (caller.empty()) return callee.str();
  if (callee.empty() || caller.ends_with(";")) return caller.str();
  return absl::StrCat(ToStringView(caller), "/", ToStringView(callee));
}

// Returns the scope a call site's caller location names.
std::string GetCallerScope(Location loc) {
  loc = SkipOpTypeWrappers(loc);
  if (auto name_loc = dyn_cast<NameLoc>(loc)) {
    return IsFrameLoc(name_loc) ? "" : GetBaseName(name_loc).str();
  }
  if (auto call_loc = dyn_cast<CallSiteLoc>(loc)) {
    return JoinScopes(GetCallerScope(call_loc.getCaller()),
                      GetCallerScope(call_loc.getCallee()));
  }
  return "";
}

// Returns the name that replaces a trailing "composite" scope for `op`.
StringRef GetCompositeName(Operation* op) {
  if (op == nullptr) return "";
  if (auto composite = dyn_cast<vhlo::CompositeOpV1>(op)) {
    if (auto name = dyn_cast<vhlo::StringV1Attr>(composite.getName())) {
      return name.getValue();
    }
  }
  if (auto custom = dyn_cast<CustomOp>(op)) return custom.getCustomCode();
  return op->getName().stripDialect();
}

std::string ReplaceCompositeScope(StringRef scope, StringRef composite_name) {
  constexpr StringRef kComposite = "composite";
  if (composite_name.empty()) return scope.str();
  if (scope == kComposite) return composite_name.str();
  if (scope.ends_with(absl::StrCat("/", ToStringView(kComposite)))) {
    return absl::StrCat(ToStringView(scope.drop_back(kComposite.size())),
                        ToStringView(composite_name));
  }
  return scope.str();
}

// Appends the names of the leaf locations of `loc` to `names`.
void CollectLeafNames(Location loc, StringRef caller_scope,
                      StringRef composite_name,
                      llvm::SmallVectorImpl<std::string>& names) {
  loc = SkipOpTypeWrappers(loc);
  if (auto fused_loc = dyn_cast<FusedLoc>(loc)) {
    for (Location sub_loc : fused_loc.getLocations()) {
      CollectLeafNames(sub_loc, caller_scope, composite_name, names);
    }
    return;
  }
  if (auto call_loc = dyn_cast<CallSiteLoc>(loc)) {
    const std::string scope =
        JoinScopes(caller_scope, GetCallerScope(call_loc.getCaller()));
    CollectLeafNames(call_loc.getCallee(), scope, composite_name, names);
    return;
  }
  StringRef leaf_name;
  if (auto name_loc = dyn_cast<NameLoc>(loc)) leaf_name = GetBaseName(name_loc);
  names.push_back(ReplaceCompositeScope(JoinScopes(caller_scope, leaf_name),
                                        composite_name));
}

// Returns true if any NameLoc in `loc` contains ';', the hierarchy delimiter
// of PyTorch locations.
bool IsPytorchLoc(Location loc) {
  if (auto name_loc = dyn_cast<NameLoc>(loc)) {
    return name_loc.getName().strref().contains(';') ||
           IsPytorchLoc(name_loc.getChildLoc());
  }
  if (auto call_loc = dyn_cast<CallSiteLoc>(loc)) {
    return IsPytorchLoc(call_loc.getCallee()) ||
           IsPytorchLoc(call_loc.getCaller());
  }
  if (auto fused_loc = dyn_cast<FusedLoc>(loc)) {
    return llvm::any_of(fused_loc.getLocations(), IsPytorchLoc);
  }
  return false;
}

StringRef GetDirectory(StringRef name) {
  const size_t last_slash = name.rfind('/');
  if (last_slash == StringRef::npos || last_slash == 0) return "";
  return name.take_front(last_slash);
}

// Keeps the names in the most common directory. CSE can fuse the locations of
// identical ops from different scopes (e.g. layers); the outliers would make
// the name misleading.
void DropOutlierScopes(llvm::SmallVectorImpl<std::string>& names) {
  llvm::SmallVector<std::pair<StringRef, int>, 4> directory_counts;
  for (const std::string& name : names) {
    const StringRef directory = GetDirectory(name);
    if (directory.empty()) continue;
    auto it = llvm::find_if(directory_counts, [&](const auto& entry) {
      return entry.first == directory;
    });
    if (it == directory_counts.end()) {
      directory_counts.emplace_back(directory, 1);
    } else {
      ++it->second;
    }
  }
  if (directory_counts.size() <= 1) return;

  // The first most common directory wins ties.
  StringRef primary = directory_counts.front().first;
  int primary_count = directory_counts.front().second;
  for (const auto& [directory, count] : directory_counts) {
    if (count > primary_count) {
      primary = directory;
      primary_count = count;
    }
  }
  llvm::SmallVector<std::string, 8> kept;
  for (const std::string& name : names) {
    if (GetDirectory(name) == primary) kept.push_back(name);
  }
  names = std::move(kept);
}

}  // namespace

std::string GetHierarchicalName(Location loc, Operation* op) {
  llvm::SmallVector<std::string, 8> names;
  CollectLeafNames(loc, /*caller_scope=*/"", GetCompositeName(op), names);
  if (llvm::all_of(names, [](const std::string& n) { return n.empty(); })) {
    return "";
  }
  const bool is_pytorch = IsPytorchLoc(loc);
  if (!is_pytorch && names.size() > 1) DropOutlierScopes(names);
  std::string joined = absl::StrJoin(names, ";");
  // ',' separates names in `tf.entry_function`.
  if (!is_pytorch) absl::c_replace(joined, ',', '_');
  return joined;
}

bool PrefersHierarchicalName(Value value, StringRef signature_name) {
  if (signature_name.contains('/')) return false;
  Operation* op = value.getDefiningOp();
  if (op == nullptr || IsPytorchLoc(op->getLoc())) return false;
  return StringRef(GetHierarchicalName(op->getLoc(), op)).trim().contains('/');
}

void TensorNamer::SetName(Value value, StringRef name) {
  auto [it, inserted] = name_counts_.try_emplace(name, 0);
  ++it->second;
  names_.try_emplace(value, it->first());
}

StringRef TensorNamer::GetName(Value value) {
  if (auto it = names_.find(value); it != names_.end()) return it->second;

  std::string name = GetHierarchicalName(value.getLoc(), value.getDefiningOp());
  if (name.empty()) {
    if (auto result = dyn_cast<OpResult>(value)) {
      name = result.getOwner()->getName().getStringRef().str();
      if (result.getResultNumber() > 0) {
        absl::StrAppend(&name, ":", result.getResultNumber());
      }
    } else {
      name = absl::StrCat("arg", cast<BlockArgument>(value).getArgNumber());
    }
  }
  const StringRef unique_name = MakeUnique(name);
  names_[value] = unique_name;
  return unique_name;
}

StringRef TensorNamer::GetName(Value value, Value source) {
  if (auto it = names_.find(value); it != names_.end()) return it->second;
  auto source_it = names_.find(source);
  const StringRef name = source_it == names_.end()
                             ? GetName(source)
                             : MakeUnique(source_it->second);
  names_[value] = name;
  return name;
}

StringRef TensorNamer::MakeUnique(StringRef name) {
  auto [it, inserted] = name_counts_.try_emplace(name, 1);
  if (inserted) return it->first();
  // Append the smallest suffix, from the number of uses of `name` on, that
  // gives a new name.
  while (true) {
    const std::string candidate =
        absl::StrCat(ToStringView(name), name_counts_[name]++);
    auto [candidate_it, candidate_inserted] =
        name_counts_.try_emplace(candidate, 1);
    if (candidate_inserted) return candidate_it->first();
  }
}

}  // namespace mlir::TFL::fb_export
