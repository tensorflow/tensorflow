/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <optional>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/import/passes.h"  // IWYU pragma: keep

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTDCEPASS
#include "xla/python/ifrt/ir/import/passes.h.inc"

namespace {

// Returns true if `callee` has a single block containing only `func.return`.
bool IsEmptyCallee(mlir::func::FuncOp callee) {
  return callee.getBody().hasOneBlock() &&
         llvm::all_of(callee.getBody().front().getOperations(),
                      [](const mlir::Operation& op) {
                        return llvm::isa<mlir::func::ReturnOp>(&op);
                      });
}

// Filters an array of per-argument or per-result dictionary attributes (e.g.,
// `arg_attrs` or `res_attrs`) by dropping entries whose index is set in
// `erase`. Returns nullptr if `dict_array` is null, all entries are erased, or
// no kept entry is a non-empty `DictionaryAttr`.
mlir::ArrayAttr FilterDictArrayAttr(mlir::ArrayAttr dict_array,
                                    const llvm::BitVector& erase) {
  if (dict_array == nullptr) {
    return nullptr;
  }
  llvm::SmallVector<mlir::Attribute> kept;
  kept.reserve(dict_array.size());
  bool has_non_empty = false;
  for (auto [idx, attr] : llvm::enumerate(dict_array)) {
    if (idx < erase.size() && erase.test(idx)) {
      continue;
    }
    kept.push_back(attr);
    if (auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
        dict != nullptr && !dict.empty()) {
      has_non_empty = true;
    }
  }
  if (kept.empty() || !has_non_empty) {
    return nullptr;
  }
  return mlir::ArrayAttr::get(dict_array.getContext(), kept);
}

// Partition of an `IfrtArrayOpInterface` op's array outputs into used (kept)
// and unused (erased) results, along with index remapping and kept types.
struct UsedOutputsSlice {
  llvm::BitVector keep;
  llvm::BitVector erase;
  llvm::SmallVector<int32_t> old_to_new;
  llvm::SmallVector<mlir::Type> kept_types;
};

// Inspects the array outputs of `op` and returns a `UsedOutputsSlice` recording
// which outputs are used vs. unused, their compacted output indices, and the
// types of the kept outputs.
UsedOutputsSlice CollectUsedOutputs(IfrtArrayOpInterface op) {
  mlir::ResultRange outputs = op.getOutputs();
  const unsigned num_outputs = outputs.size();
  UsedOutputsSlice slice{llvm::BitVector(num_outputs),
                         llvm::BitVector(num_outputs),
                         llvm::SmallVector<int32_t>(num_outputs, -1),
                         {}};
  slice.kept_types.reserve(num_outputs);
  for (unsigned i = 0; i < num_outputs; ++i) {
    if (outputs[i].use_empty()) {
      slice.erase.set(i);
    } else {
      slice.keep.set(i);
      slice.old_to_new[i] = slice.kept_types.size();
      slice.kept_types.push_back(outputs[i].getType());
    }
  }
  return slice;
}

// Replaces uses of kept array outputs of `old_op` with the corresponding
// compacted outputs of `new_op` according to `old_to_new` (-1 for erased
// outputs), forwards `old_op`'s control output to `new_op`'s control output,
// and erases `old_op`.
void ReplaceKeptOutputs(mlir::RewriterBase& rewriter,
                        IfrtArrayOpInterface old_op,
                        IfrtArrayOpInterface new_op,
                        llvm::ArrayRef<int32_t> old_to_new) {
  for (auto [i, new_idx] : llvm::enumerate(old_to_new)) {
    if (new_idx != -1) {
      rewriter.replaceAllUsesWith(old_op.getOutputs()[i],
                                  new_op.getOutputs()[new_idx]);
    }
  }
  rewriter.replaceAllUsesWith(old_op.getControlOutput(),
                              new_op.getControlOutput());
  rewriter.eraseOp(old_op);
}

// Returns true if `control_inputs` can be forwarded to all users of
// `control_output` via `detail::ForwardControlInputsToUsers` (either all users
// accept variadic IFRT control dependencies, or there is a single unique
// control input).
bool CanForwardControlInputs(mlir::Value control_output,
                             mlir::ValueRange control_inputs) {
  if (llvm::all_of(control_output.getUsers(), [](mlir::Operation* user) {
        return llvm::isa<IfrtArrayOpInterface, AssembleOp, DisassembleOp>(user);
      })) {
    return true;
  }
  llvm::SmallDenseSet<mlir::Value, 4> unique(control_inputs.begin(),
                                             control_inputs.end());
  return unique.size() == 1;
}

// Eliminates unused `(input, output)` pairs from `op`, or erases `op` entirely
// (forwarding its `control_inputs` to consumers of its `control_output`) if all
// array outputs are unused.
//
// REQUIRES: `op` is side-effect free.
void EliminateUnusedPairwiseArrayOpOutputs(IfrtPairwiseArrayOpInterface op,
                                           mlir::RewriterBase& rewriter) {
  UsedOutputsSlice slice = CollectUsedOutputs(op);
  if (slice.keep.none() &&
      !CanForwardControlInputs(op.getControlOutput(), op.getControlInputs())) {
    return;
  }
  op.replaceWithKeptOutputs(rewriter, slice.keep);
}

// Eliminates unused array outputs from `remap_op` (or erases `remap_op`
// entirely and forwards its `control_inputs` to consumers of its
// `control_output` if no array outputs are used), pruning unreferenced input
// arrays and compacting `in_array_index` and `out_array_index` in kept
// `IfrtArrayMappingAttr` entries.
void EliminateUnusedRemapArraysOutputs(RemapArraysOp remap_op,
                                       mlir::RewriterBase& rewriter) {
  UsedOutputsSlice slice = CollectUsedOutputs(remap_op);
  if (slice.keep.none()) {
    if (CanForwardControlInputs(remap_op.getControlOutput(),
                                remap_op.getControlInputs())) {
      detail::ForwardControlInputsToUsers(
          remap_op.getControlOutput(), remap_op.getControlInputs(), rewriter);
      rewriter.eraseOp(remap_op);
    }
    return;
  }

  const unsigned num_inputs = remap_op.getInputs().size();
  llvm::BitVector used_inputs(num_inputs);
  llvm::SmallVector<IfrtArrayMappingAttr> kept_mappings;
  kept_mappings.reserve(remap_op.getMappings().size());
  for (mlir::Attribute attr : remap_op.getMappings()) {
    auto mapping = llvm::cast<IfrtArrayMappingAttr>(attr);
    if (slice.old_to_new[mapping.getOutArrayIndex()] == -1) {
      continue;
    }
    used_inputs.set(mapping.getInArrayIndex());
    kept_mappings.push_back(mapping);
  }

  if (slice.erase.none() && used_inputs.all()) {
    return;
  }

  llvm::SmallVector<int32_t> old_in_to_new_in(num_inputs, -1);
  llvm::SmallVector<mlir::Value> new_inputs;
  new_inputs.reserve(num_inputs);
  for (unsigned i = 0; i < num_inputs; ++i) {
    if (used_inputs.test(i)) {
      old_in_to_new_in[i] = new_inputs.size();
      new_inputs.push_back(remap_op.getInputs()[i]);
    }
  }

  if (new_inputs.empty() || kept_mappings.empty()) {
    return;
  }

  llvm::SmallVector<mlir::Attribute> new_mappings;
  new_mappings.reserve(kept_mappings.size());
  for (IfrtArrayMappingAttr mapping : kept_mappings) {
    new_mappings.push_back(IfrtArrayMappingAttr::get(
        rewriter.getContext(), old_in_to_new_in[mapping.getInArrayIndex()],
        slice.old_to_new[mapping.getOutArrayIndex()], mapping.getMappings()));
  }

  rewriter.setInsertionPoint(remap_op);
  RemapArraysOp new_remap = RemapArraysOp::create(
      rewriter, remap_op.getLoc(),
      /*outputs=*/slice.kept_types,
      /*control_output=*/remap_op.getControlOutput().getType(),
      /*inputs=*/new_inputs,
      /*mappings=*/rewriter.getArrayAttr(new_mappings),
      /*control_inputs=*/remap_op.getControlInputs(),
      /*donated=*/remap_op.getDonatedAttr());
  new_remap->setDiscardableAttrs(remap_op->getDiscardableAttrDictionary());

  ReplaceKeptOutputs(rewriter, remap_op, new_remap, slice.old_to_new);
}

// Resolved target of an `ifrt.Call` callee symbol reference, supporting either
// a top-level `func.func` or a `func.func` nested inside a `builtin.module`.
struct ResolvedCallee {
  mlir::Operation* root_symbol_op = nullptr;
  mlir::StringAttr root_symbol_name;
  mlir::func::FuncOp func_op;
  bool is_nested_module = false;
};

// Resolves `call_op.getCallee()` in `symbol_table` to a non-external
// `func.func`, supporting either a flat symbol ref (`@callee`) or a 1-level
// nested module ref (`@module::@callee`).
std::optional<ResolvedCallee> ResolveCallee(CallOp call_op,
                                            mlir::SymbolTable& symbol_table) {
  mlir::SymbolRefAttr callee_ref = call_op.getCallee();
  mlir::StringAttr root_name = callee_ref.getRootReference();
  mlir::Operation* root_op = symbol_table.lookup(root_name);
  auto func_op = llvm::dyn_cast_or_null<mlir::func::FuncOp>(root_op);
  if (callee_ref.getNestedReferences().size() == 1) {
    if (auto mod = llvm::dyn_cast_or_null<mlir::ModuleOp>(root_op);
        mod != nullptr) {
      func_op =
          mod.lookupSymbol<mlir::func::FuncOp>(callee_ref.getLeafReference());
    }
  } else if (!callee_ref.getNestedReferences().empty()) {
    return std::nullopt;
  }
  if (func_op == nullptr || func_op.isExternal()) {
    return std::nullopt;
  }
  return ResolvedCallee{root_op, root_name, func_op, root_op != func_op};
}

// Eliminates unused results and inputs of `call_op` and its resolved callee:
// clones a private copy of the callee (or enclosing module) when shared,
// erases unused `call_op` outputs and callee `func.return` operands, simplifies
// the callee region, erases unused callee arguments and `call_op` inputs while
// remapping aliasing/donation/arg/result attributes, and erases `call_op` (plus
// its private callee if unused) when it becomes an empty dead call.
// NOTE: `call_op` is not removed if it is not side-effect free.
void EliminateCallDce(CallOp call_op, mlir::ModuleOp module_op,
                      mlir::SymbolTable& symbol_table,
                      llvm::DenseMap<mlir::StringAttr, int>& symbol_use_counts,
                      bool symbol_uses_known, mlir::RewriterBase& rewriter) {
  std::optional<ResolvedCallee> resolved = ResolveCallee(call_op, symbol_table);
  if (!resolved.has_value()) {
    return;
  }

  const int root_uses = symbol_use_counts.lookup(resolved->root_symbol_name);
  const bool is_shared = !symbol_uses_known || root_uses != 1;

  mlir::Operation* root_op = resolved->root_symbol_op;
  mlir::func::FuncOp callee = resolved->func_op;
  mlir::SymbolRefAttr callee_ref = call_op.getCalleeAttr();
  mlir::StringAttr cloned_root_name;
  if (is_shared) {
    root_op = resolved->root_symbol_op->clone();
    mlir::SymbolTable::setSymbolVisibility(
        root_op, mlir::SymbolTable::Visibility::Private);
    cloned_root_name = symbol_table.insert(
        root_op,
        mlir::Block::iterator(resolved->root_symbol_op->getNextNode()));
    if (resolved->is_nested_module) {
      auto cloned_module = llvm::cast<mlir::ModuleOp>(root_op);
      mlir::StringAttr leaf_name = call_op.getCallee().getLeafReference();
      callee =
          llvm::cast<mlir::func::FuncOp>(cloned_module.lookupSymbol(leaf_name));
      callee_ref = mlir::SymbolRefAttr::get(
          cloned_root_name, {mlir::SymbolRefAttr::get(leaf_name)});
    } else {
      callee = llvm::cast<mlir::func::FuncOp>(root_op);
      callee_ref = mlir::FlatSymbolRefAttr::get(cloned_root_name);
    }
  }

  // 1. Eliminate unused CallOp results and corresponding callee return values.
  UsedOutputsSlice out_slice = CollectUsedOutputs(call_op);
  if (out_slice.erase.any()) {
    for (mlir::Block& block : callee.getBody()) {
      if (auto return_op =
              llvm::dyn_cast<mlir::func::ReturnOp>(block.getTerminator());
          return_op != nullptr) {
        return_op->eraseOperands(out_slice.erase);
      }
    }
    CHECK(mlir::succeeded(callee.eraseResults(out_slice.erase)));
  }

  // 2. Simplify the callee region so pure ops feeding only erased return values
  // are eliminated before checking block argument uses.
  const bool simplified =
      mlir::succeeded(mlir::simplifyRegions(rewriter, callee.getBody()));

  // 3. Eliminate unused callee entry block arguments and CallOp inputs.
  mlir::Block& entry_block = callee.getBody().front();
  const unsigned num_inputs = call_op.getInputs().size();
  llvm::BitVector unused_arguments(num_inputs);
  llvm::SmallVector<int32_t> old_in_to_new_in(num_inputs, -1);
  llvm::SmallVector<mlir::Value> new_inputs;
  new_inputs.reserve(num_inputs);
  for (mlir::BlockArgument arg : entry_block.getArguments()) {
    const unsigned idx = arg.getArgNumber();
    if (arg.use_empty()) {
      unused_arguments.set(idx);
    } else {
      old_in_to_new_in[idx] = new_inputs.size();
      new_inputs.push_back(call_op.getInputs()[idx]);
    }
  }

  if (out_slice.erase.any()) {
    for (unsigned old_in = 0; old_in < num_inputs; ++old_in) {
      if (unused_arguments.test(old_in)) {
        continue;
      }
      if (auto aliasing_attr = callee.getArgAttrOfType<mlir::IntegerAttr>(
              old_in, kAliasingOutputAttrName);
          aliasing_attr != nullptr) {
        if (const int64_t old_out = aliasing_attr.getInt();
            out_slice.old_to_new[old_out] == -1) {
          callee.removeArgAttr(old_in, kAliasingOutputAttrName);
        } else if (out_slice.old_to_new[old_out] != old_out) {
          callee.setArgAttr(
              old_in, kAliasingOutputAttrName,
              mlir::IntegerAttr::get(aliasing_attr.getType(),
                                     out_slice.old_to_new[old_out]));
        }
      }
    }
  }

  if (unused_arguments.any()) {
    CHECK(mlir::succeeded(callee.eraseArguments(unused_arguments)));
  }

  // Remap donated_input_indices and io_aliases.
  llvm::SmallVector<int32_t> new_donated_input_indices;
  for (int32_t old_in : call_op.getDonatedInputIndices()) {
    if (old_in_to_new_in[old_in] != -1) {
      new_donated_input_indices.push_back(old_in_to_new_in[old_in]);
    }
  }
  llvm::SmallVector<mlir::Attribute> new_io_aliases;
  for (mlir::DenseI32ArrayAttr raw_alias :
       call_op.getIoAliases().getAsRange<mlir::DenseI32ArrayAttr>()) {
    llvm::ArrayRef<int32_t> pair = raw_alias.asArrayRef();
    const int32_t old_in = pair[0];
    const int32_t old_out = pair[1];
    if (old_in_to_new_in[old_in] != -1 && out_slice.old_to_new[old_out] != -1) {
      new_io_aliases.push_back(rewriter.getDenseI32ArrayAttr(
          {old_in_to_new_in[old_in], out_slice.old_to_new[old_out]}));
    }
  }

  // 4. Erase empty dead CallOp and its unused private callee.
  if (out_slice.kept_types.empty() && new_inputs.empty() &&
      IsEmptyCallee(callee) &&
      CanForwardControlInputs(call_op.getControlOutput(),
                              call_op.getControlInputs())) {
    detail::ForwardControlInputsToUsers(call_op.getControlOutput(),
                                        call_op.getControlInputs(), rewriter);
    rewriter.eraseOp(call_op);
    int& remaining_uses = symbol_use_counts[resolved->root_symbol_name];
    --remaining_uses;
    if (is_shared) {
      symbol_table.erase(root_op);
    }
    if (mlir::SymbolTable::getSymbolVisibility(resolved->root_symbol_op) ==
            mlir::SymbolTable::Visibility::Private &&
        ((symbol_uses_known && remaining_uses == 0) ||
         mlir::SymbolTable::symbolKnownUseEmpty(resolved->root_symbol_op,
                                                module_op))) {
      symbol_table.erase(resolved->root_symbol_op);
    }
    return;
  }

  if (out_slice.erase.any() || unused_arguments.any()) {
    mlir::ArrayAttr new_arg_attrs =
        FilterDictArrayAttr(call_op.getArgAttrsAttr(), unused_arguments);
    mlir::ArrayAttr new_res_attrs =
        FilterDictArrayAttr(call_op.getResAttrsAttr(), out_slice.erase);

    rewriter.setInsertionPoint(call_op);
    CallOp new_call = CallOp::create(
        rewriter, call_op.getLoc(),
        /*outputs=*/out_slice.kept_types,
        /*control_output=*/call_op.getControlOutput().getType(),
        /*inputs=*/new_inputs,
        /*control_inputs=*/call_op.getControlInputs(),
        /*arg_attrs=*/new_arg_attrs,
        /*res_attrs=*/new_res_attrs,
        /*callee=*/callee_ref,
        /*devices=*/call_op.getDevicesAttr(),
        /*io_aliases=*/rewriter.getArrayAttr(new_io_aliases),
        /*donated_input_indices=*/
        rewriter.getDenseI32ArrayAttr(new_donated_input_indices));
    new_call->setDiscardableAttrs(call_op->getDiscardableAttrDictionary());

    ReplaceKeptOutputs(rewriter, call_op, new_call, out_slice.old_to_new);
    if (is_shared) {
      --symbol_use_counts[resolved->root_symbol_name];
      symbol_use_counts[cloned_root_name] = 1;
    }
    return;
  }

  if (is_shared) {
    if (simplified) {
      call_op.setCalleeAttr(callee_ref);
      --symbol_use_counts[resolved->root_symbol_name];
      symbol_use_counts[cloned_root_name] = 1;
    } else {
      symbol_table.erase(root_op);
    }
  }
}

class IfrtDcePass : public impl::IfrtDcePassBase<IfrtDcePass> {
 public:
  using impl::IfrtDcePassBase<IfrtDcePass>::IfrtDcePassBase;

  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    mlir::IRRewriter rewriter(&getContext());
    mlir::SymbolTable symbol_table(module_op);

    llvm::DenseMap<mlir::StringAttr, int> symbol_use_counts;
    bool symbol_uses_known = false;
    if (std::optional<mlir::SymbolTable::UseRange> uses =
            mlir::SymbolTable::getSymbolUses(&module_op.getBodyRegion());
        uses.has_value()) {
      symbol_uses_known = true;
      for (const mlir::SymbolTable::SymbolUse& use : *uses) {
        ++symbol_use_counts[use.getSymbolRef().getRootReference()];
      }
    }

    // Snapshot IFRT entry functions upfront because `EliminateCallDce` inserts
    // and erases private callee `func.func` ops in `module_op`, which would
    // invalidate a live `module_op.getOps<mlir::func::FuncOp>()` iterator.
    llvm::SmallVector<mlir::func::FuncOp> ifrt_funcs;
    for (mlir::func::FuncOp func_op : module_op.getOps<mlir::func::FuncOp>()) {
      if (IsIfrtFunction(func_op)) {
        ifrt_funcs.push_back(func_op);
      }
    }

    for (mlir::func::FuncOp func_op : ifrt_funcs) {
      func_op.walk<mlir::WalkOrder::PreOrder, mlir::ReverseIterator>(
          [&](mlir::Operation* op) {
            return llvm::TypeSwitch<mlir::Operation*, mlir::WalkResult>(op)
                .Case<CallOp>([&](CallOp call_op) {
                  EliminateCallDce(call_op, module_op, symbol_table,
                                   symbol_use_counts, symbol_uses_known,
                                   rewriter);
                  return mlir::WalkResult::skip();
                })
                .Case<IfrtPairwiseArrayOpInterface>(
                    [&](IfrtPairwiseArrayOpInterface pairwise_op) {
                      EliminateUnusedPairwiseArrayOpOutputs(pairwise_op,
                                                            rewriter);
                      return mlir::WalkResult::skip();
                    })
                .Case<RemapArraysOp>([&](RemapArraysOp remap_op) {
                  EliminateUnusedRemapArraysOutputs(remap_op, rewriter);
                  return mlir::WalkResult::skip();
                })
                .Default(mlir::WalkResult::advance());
          });
    }
  }
};

}  // namespace
}  // namespace ifrt
}  // namespace xla
