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

#include "tensorflow/compiler/mlir/lite/python/location_passes.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/utils/name_utils.h"

namespace mlir::TFL {
namespace {

absl::string_view ToAbslStringView(llvm::StringRef ref) {
  return absl::string_view(ref.data(), ref.size());
}

// Returns true if `loc` represents a Python traceback frame NameLoc, i.e., a
// NameLoc whose name is a non-hierarchical function/node identifier (no `/`)
// and whose immediate child is a FileLineColLoc (as emitted by JAX's
// traceback_to_location).
bool IsFrameNameLoc(mlir::Location loc) {
  auto name_loc = mlir::dyn_cast<mlir::NameLoc>(loc);
  return name_loc && mlir::isa<mlir::FileLineColLoc>(name_loc.getChildLoc()) &&
         !name_loc.getName().strref().contains('/');
}

// Unwraps leading JAX op-type NameLoc wrappers (whose name ends with ':', e.g.
// "composite:", "pjit:", "dot_general:") so the underlying scope NameLoc is
// visible to `mlir::GetNameFromLoc`.
mlir::Location StripOpTypeWrappers(mlir::Location loc) {
  while (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
    if (IsFrameNameLoc(name_loc)) {
      break;
    }
    llvm::StringRef name = name_loc.getName().strref().split('@').first;
    if (!name.ends_with(":")) {
      break;
    }
    loc = name_loc.getChildLoc();
  }
  return loc;
}

struct ParsedScopeLoc {
  llvm::StringRef scope;
  mlir::Location base_loc;
};

ParsedScopeLoc ParseScopeLoc(mlir::Location loc) {
  if (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(loc);
      name_loc && !IsFrameNameLoc(name_loc)) {
    return {name_loc.getName().strref(), name_loc.getChildLoc()};
  }
  return {llvm::StringRef(), loc};
}

// Strips JAX's internal `@api.jit def _where` wrapper (`jit(_where)`) when the
// callee already carries an op name (such as `select_n`), so multiple
// `jnp.where` calls in the same module scope stay distinct leaf ops rather
// than grouping into a shared `jit(_where)` sub-scope.
llvm::StringRef StripInternalJitWrapper(llvm::StringRef caller_scope,
                                        llvm::StringRef callee_scope) {
  if (callee_scope.empty()) {
    return caller_scope;
  }
  constexpr llvm::StringLiteral kJitWhere = "jit(_where)";
  constexpr llvm::StringLiteral kSlashJitWhere = "/jit(_where)";
  if (caller_scope == kJitWhere) {
    return llvm::StringRef();
  }
  if (caller_scope.ends_with(kSlashJitWhere)) {
    return caller_scope.drop_back(kSlashJitWhere.size());
  }
  return caller_scope;
}

mlir::Location NormalizeLocation(
    mlir::Location loc, mlir::MLIRContext* context,
    llvm::DenseMap<mlir::Location, mlir::Location>& memo) {
  if (auto it = memo.find(loc); it != memo.end()) {
    return it->second;
  }

  mlir::Location unwrapped = StripOpTypeWrappers(loc);
  mlir::Location result = unwrapped;

  if (auto call_loc = mlir::dyn_cast<mlir::CallSiteLoc>(unwrapped)) {
    mlir::Location norm_callee =
        NormalizeLocation(call_loc.getCallee(), context, memo);
    mlir::Location norm_caller =
        NormalizeLocation(call_loc.getCaller(), context, memo);
    ParsedScopeLoc caller_info = ParseScopeLoc(norm_caller);
    ParsedScopeLoc callee_info = ParseScopeLoc(norm_callee);

    if (!caller_info.scope.empty()) {
      llvm::StringRef caller_scope =
          StripInternalJitWrapper(caller_info.scope, callee_info.scope);
      std::string combined_scope;
      if (caller_scope.empty()) {
        combined_scope = std::string(callee_info.scope);
      } else if (callee_info.scope.empty() || caller_scope.ends_with(";")) {
        combined_scope = std::string(caller_scope);
      } else {
        combined_scope = absl::StrCat(ToAbslStringView(caller_scope), "/",
                                      ToAbslStringView(callee_info.scope));
      }

      mlir::Location combined_base = callee_info.base_loc;
      if (mlir::isa<mlir::UnknownLoc>(combined_base)) {
        combined_base = caller_info.base_loc;
      } else if (!mlir::isa<mlir::UnknownLoc>(caller_info.base_loc)) {
        combined_base =
            mlir::CallSiteLoc::get(callee_info.base_loc, caller_info.base_loc);
      }
      result = mlir::NameLoc::get(
          mlir::StringAttr::get(context, combined_scope), combined_base);
    } else if (norm_callee != call_loc.getCallee() ||
               norm_caller != call_loc.getCaller()) {
      result = mlir::CallSiteLoc::get(norm_callee, norm_caller);
    }
  } else if (auto fused_loc = mlir::dyn_cast<mlir::FusedLoc>(unwrapped)) {
    llvm::SmallVector<mlir::Location, 4> new_locs;
    new_locs.reserve(fused_loc.getLocations().size());
    bool changed = false;
    for (mlir::Location sub_loc : fused_loc.getLocations()) {
      mlir::Location norm_sub = NormalizeLocation(sub_loc, context, memo);
      if (norm_sub != sub_loc) {
        changed = true;
      }
      new_locs.push_back(norm_sub);
    }
    if (changed) {
      result = mlir::FusedLoc::get(context, new_locs, fused_loc.getMetadata());
    }
  }

  memo.try_emplace(loc, result);
  return result;
}

// Replaces generic JAX `"composite"` / `".../composite"` leaf names on
// `stablehlo.composite` ops with the composite's semantic `name` attribute
// (e.g. `odml.runtime_bmm`, `odml.cache_update`).
mlir::Location EnrichCompositeLocation(
    mlir::stablehlo::CompositeOp composite_op, mlir::Location loc) {
  llvm::StringRef composite_name = composite_op.getName();
  if (composite_name.empty()) {
    return loc;
  }
  auto name_loc = mlir::dyn_cast<mlir::NameLoc>(loc);
  if (!name_loc || IsFrameNameLoc(name_loc)) {
    return loc;
  }
  llvm::StringRef name = name_loc.getName().strref();
  constexpr llvm::StringLiteral kComposite = "composite";
  constexpr llvm::StringLiteral kSlashComposite = "/composite";
  std::string enriched_name;
  if (name == kComposite) {
    enriched_name = std::string(composite_name);
  } else if (name.ends_with(kSlashComposite)) {
    llvm::StringRef prefix = name.drop_back(kComposite.size());
    enriched_name = absl::StrCat(ToAbslStringView(prefix),
                                 ToAbslStringView(composite_name));
  } else {
    return loc;
  }
  return mlir::NameLoc::get(
      mlir::StringAttr::get(composite_op.getContext(), enriched_name),
      name_loc.getChildLoc());
}

class NormalizeLocationsPass
    : public mlir::PassWrapper<NormalizeLocationsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeLocationsPass)

  llvm::StringRef getArgument() const final {
    return "tfl-normalize-locations";
  }
  llvm::StringRef getDescription() const final {
    return "Normalizes JAX op-type NameLocs, inlined CallSiteLoc scopes, and "
           "StableHLO composite leaf names for hierarchy preservation.";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = module.getContext();
    llvm::DenseMap<mlir::Location, mlir::Location> memo;

    module.walk([&](mlir::Operation* op) {
      mlir::Location old_loc = op->getLoc();
      mlir::Location new_loc = NormalizeLocation(old_loc, context, memo);
      if (auto composite_op =
              mlir::dyn_cast<mlir::stablehlo::CompositeOp>(op)) {
        new_loc = EnrichCompositeLocation(composite_op, new_loc);
      }
      if (new_loc != old_loc) {
        op->setLoc(new_loc);
      }
    });
  }
};

// Returns true if any `NameLoc` in `loc` contains `;` (the PyTorch
// hierarchy-delimiter convention emitted by `litert_torch`), so JAX-specific
// location cleanup leaves PyTorch locations untouched.
bool HasSemicolonNameLoc(mlir::Location loc) {
  if (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
    if (name_loc.getName().strref().contains(';')) {
      return true;
    }
    return HasSemicolonNameLoc(name_loc.getChildLoc());
  }
  if (auto call_loc = mlir::dyn_cast<mlir::CallSiteLoc>(loc)) {
    return HasSemicolonNameLoc(call_loc.getCallee()) ||
           HasSemicolonNameLoc(call_loc.getCaller());
  }
  if (auto fused_loc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location sub_loc : fused_loc.getLocations()) {
      if (HasSemicolonNameLoc(sub_loc)) {
        return true;
      }
    }
  }
  return false;
}

void FlattenFusedLoc(mlir::Location loc,
                     llvm::SmallVectorImpl<mlir::Location>& out) {
  if (auto fused_loc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location sub_loc : fused_loc.getLocations()) {
      FlattenFusedLoc(sub_loc, out);
    }
    return;
  }
  out.push_back(loc);
}

// Returns the parent directory scope (the prefix before the last `/`) of a
// scoped `NameLoc`, or an empty `StringRef` if `loc` has no hierarchical scope.
llvm::StringRef ExtractDirectoryScope(mlir::Location loc) {
  mlir::Location cur = StripOpTypeWrappers(loc);
  while (auto call_loc = mlir::dyn_cast<mlir::CallSiteLoc>(cur)) {
    cur = StripOpTypeWrappers(call_loc.getCallee());
  }
  auto name_loc = mlir::dyn_cast<mlir::NameLoc>(cur);
  if (!name_loc || IsFrameNameLoc(name_loc)) {
    return llvm::StringRef();
  }
  llvm::StringRef name = name_loc.getName().strref().split('@').first;
  size_t last_slash = name.rfind('/');
  if (last_slash == llvm::StringRef::npos || last_slash == 0) {
    return llvm::StringRef();
  }
  return name.substr(0, last_slash);
}

// Removes cross-scope CSE outlier locations from `FusedLoc`s. When a rewrite
// pattern fuses a root op in layer `i` with a sub-expression that CSE hoisted
// into layer `0`, the resulting `FusedLoc` mixes directory scopes (with the
// root replaced op first or in the majority). Keeping only locations in the
// primary directory scope prevents backward location heuristics from assigning
// the fused op to layer `0`.
mlir::Location SanitizeCrossScopeFusedLocation(mlir::Location loc,
                                               mlir::MLIRContext* context) {
  auto fused_loc = mlir::dyn_cast<mlir::FusedLoc>(loc);
  if (!fused_loc || HasSemicolonNameLoc(loc)) {
    return loc;
  }

  llvm::SmallVector<mlir::Location, 8> flat_locs;
  FlattenFusedLoc(loc, flat_locs);
  if (flat_locs.size() <= 1) {
    return loc;
  }

  llvm::SmallVector<std::pair<llvm::StringRef, int>, 4> dir_counts;
  llvm::SmallVector<llvm::StringRef, 8> elem_dirs;
  elem_dirs.reserve(flat_locs.size());
  for (mlir::Location sub_loc : flat_locs) {
    llvm::StringRef dir = ExtractDirectoryScope(sub_loc);
    elem_dirs.push_back(dir);
    if (dir.empty()) {
      continue;
    }
    auto it = llvm::find_if(
        dir_counts, [&](const auto& entry) { return entry.first == dir; });
    if (it != dir_counts.end()) {
      ++it->second;
    } else {
      dir_counts.emplace_back(dir, 1);
    }
  }
  if (dir_counts.size() <= 1) {
    return loc;
  }

  // Pick the directory scope with the highest frequency; ties keep the first
  // appearance (the root op replaced by TableGen/pattern rewrites).
  llvm::StringRef primary_dir = dir_counts[0].first;
  int best_count = dir_counts[0].second;
  for (size_t i = 1; i < dir_counts.size(); ++i) {
    if (dir_counts[i].second > best_count) {
      primary_dir = dir_counts[i].first;
      best_count = dir_counts[i].second;
    }
  }

  llvm::SmallVector<mlir::Location, 8> kept_locs;
  kept_locs.reserve(flat_locs.size());
  for (size_t i = 0; i < flat_locs.size(); ++i) {
    if (elem_dirs[i] == primary_dir) {
      kept_locs.push_back(flat_locs[i]);
    }
  }
  if (kept_locs.empty() || kept_locs.size() == flat_locs.size()) {
    return loc;
  }
  if (kept_locs.size() == 1) {
    return kept_locs[0];
  }
  return mlir::FusedLoc::get(context, kept_locs, fused_loc.getMetadata());
}

// Updates `tf.entry_function` `outputs` tensor names from flat placeholder
// strings (such as `decode_kv_cache_k_0_output`) to the defining op's unique
// hierarchical location name when available, while leaving `SignatureDef`
// output names (`tf_saved_model.index_path`) unchanged.
void PreserveHierarchicalEntryOutputNames(mlir::ModuleOp module) {
  struct EntryFuncInfo {
    mlir::func::FuncOp func;
    mlir::DictionaryAttr dict_attr;
    llvm::SmallVector<llvm::StringRef, 8> raw_outputs;
    llvm::SmallVector<std::string, 8> candidate_loc_names;
  };

  llvm::SmallVector<EntryFuncInfo, 4> entry_funcs;
  absl::flat_hash_set<std::string> used_names;

  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (func.isDeclaration()) {
      continue;
    }
    auto dict_attr =
        func->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
    if (!dict_attr) {
      continue;
    }
    if (auto inputs_attr =
            mlir::dyn_cast_or_null<mlir::StringAttr>(dict_attr.get("inputs"))) {
      llvm::SmallVector<llvm::StringRef, 8> input_names;
      inputs_attr.getValue().split(input_names, ',', /*MaxSplit=*/-1,
                                   /*KeepEmpty=*/false);
      for (llvm::StringRef input_name : input_names) {
        used_names.insert(std::string(input_name.trim()));
      }
    }
    auto outputs_attr =
        mlir::dyn_cast_or_null<mlir::StringAttr>(dict_attr.get("outputs"));
    if (!outputs_attr) {
      continue;
    }
    llvm::SmallVector<llvm::StringRef, 8> output_names;
    outputs_attr.getValue().split(output_names, ',', /*MaxSplit=*/-1,
                                  /*KeepEmpty=*/false);
    mlir::Operation* terminator = func.back().getTerminator();
    if (terminator == nullptr ||
        output_names.size() != terminator->getNumOperands()) {
      for (llvm::StringRef output_name : output_names) {
        used_names.insert(std::string(output_name.trim()));
      }
      continue;
    }

    EntryFuncInfo info;
    info.func = func;
    info.dict_attr = dict_attr;
    info.raw_outputs = std::move(output_names);
    info.candidate_loc_names.resize(terminator->getNumOperands());
    bool has_candidate = false;
    for (unsigned i = 0; i < terminator->getNumOperands(); ++i) {
      mlir::Value operand = terminator->getOperand(i);
      mlir::Operation* def_op = operand.getDefiningOp();
      std::string candidate;
      if (def_op != nullptr && !HasSemicolonNameLoc(def_op->getLoc())) {
        std::string loc_name = mlir::GetNameFromLoc(def_op->getLoc());
        // `tf.entry_function` splits `outputs` on `,`, so replace commas (e.g.
        // from einsum equation scopes) before storing.
        for (char& ch : loc_name) {
          if (ch == ',') {
            ch = '_';
          }
        }
        llvm::StringRef trimmed = llvm::StringRef(loc_name).trim();
        if (trimmed.contains('/')) {
          candidate = std::string(trimmed);
        }
      }
      if (candidate.empty()) {
        used_names.insert(std::string(info.raw_outputs[i].trim()));
      } else {
        info.candidate_loc_names[i] = std::move(candidate);
        has_candidate = true;
      }
    }
    if (has_candidate) {
      entry_funcs.push_back(std::move(info));
    }
  }

  llvm::DenseMap<mlir::Value, std::string> assigned_value_names;
  for (EntryFuncInfo& info : entry_funcs) {
    mlir::Operation* terminator = info.func.back().getTerminator();
    std::vector<std::string> final_outputs;
    final_outputs.reserve(info.raw_outputs.size());
    for (size_t i = 0; i < info.raw_outputs.size(); ++i) {
      if (info.candidate_loc_names[i].empty()) {
        final_outputs.push_back(std::string(info.raw_outputs[i].trim()));
        continue;
      }
      mlir::Value operand = terminator->getOperand(i);
      if (auto it = assigned_value_names.find(operand);
          it != assigned_value_names.end()) {
        final_outputs.push_back(it->second);
        continue;
      }
      const std::string& base_name = info.candidate_loc_names[i];
      std::string unique_name = base_name;
      int suffix = 1;
      while (used_names.contains(unique_name)) {
        unique_name = absl::StrCat(base_name, suffix++);
      }
      used_names.insert(unique_name);
      assigned_value_names.try_emplace(operand, unique_name);
      final_outputs.push_back(std::move(unique_name));
    }

    llvm::SmallVector<mlir::NamedAttribute, 4> updated_attrs;
    updated_attrs.reserve(info.dict_attr.size());
    for (const mlir::NamedAttribute& named_attr : info.dict_attr) {
      if (named_attr.getName() == "outputs") {
        updated_attrs.emplace_back(
            named_attr.getName(),
            mlir::StringAttr::get(module.getContext(),
                                  absl::StrJoin(final_outputs, ",")));
      } else {
        updated_attrs.push_back(named_attr);
      }
    }
    info.func->setAttr(
        "tf.entry_function",
        mlir::DictionaryAttr::get(module.getContext(), updated_attrs));
  }
}

class FinalizeExportLocationsPass
    : public mlir::PassWrapper<FinalizeExportLocationsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeExportLocationsPass)

  llvm::StringRef getArgument() const final {
    return "tfl-finalize-export-locations";
  }
  llvm::StringRef getDescription() const final {
    return "Sanitizes cross-scope FusedLoc outliers and preserves hierarchical "
           "locations on entry-function output tensors before export.";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = module.getContext();

    module.walk([&](mlir::Operation* op) {
      mlir::Location old_loc = op->getLoc();
      mlir::Location new_loc =
          SanitizeCrossScopeFusedLocation(old_loc, context);
      if (new_loc != old_loc) {
        op->setLoc(new_loc);
      }
    });

    PreserveHierarchicalEntryOutputNames(module);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateNormalizeLocationsPass() {
  return std::make_unique<NormalizeLocationsPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateFinalizeExportLocationsPass() {
  return std::make_unique<FinalizeExportLocationsPass>();
}

}  // namespace mlir::TFL
