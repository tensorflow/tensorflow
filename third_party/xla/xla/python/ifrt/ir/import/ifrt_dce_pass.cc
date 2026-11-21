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

#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"
#include "llvm/ADT/BitVector.h"
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
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/import/callee_symbol_manager.h"
#include "xla/python/ifrt/ir/import/passes.h"  // IWYU pragma: keep
#include "xla/python/ifrt/ir/utils.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTDCEPASS
#include "xla/python/ifrt/ir/import/passes.h.inc"

namespace {

// Returns true if `callee` has a single block containing only `func.return` and
// operations that would be trivially dead once `func.return` operands are
// dropped.
bool IsSideEffectFreeCallee(mlir::func::FuncOp callee) {
  return callee.getBody().hasOneBlock() &&
         llvm::all_of(callee.getBody().front().getOperations(),
                      [](mlir::Operation& op) {
                        return llvm::isa<mlir::func::ReturnOp>(&op) ||
                               mlir::wouldOpBeTriviallyDead(&op);
                      });
}

// Eliminates unused `(input, output)` pairs from `op`, or erases `op` entirely
// (forwarding its `control_inputs` to consumers of its `control_output`) if all
// array outputs are unused.
//
// REQUIRES: `op` is side-effect free.
void EliminateUnusedPairwiseArrayOpOutputs(IfrtPairwiseArrayOpInterface op,
                                           CalleeSymbolManager& symbol_manager,
                                           mlir::RewriterBase& rewriter) {
  UsedOutputsSlice slice = CollectUsedOutputs(op);
  if (slice.keep.none()) {
    if (!CanForwardControlInputs(op.getControlOutput(),
                                 op.getControlInputs())) {
      return;
    }
    symbol_manager.RecordOpErased(op);
  }
  op.replaceWithKeptOutputs(rewriter, slice.keep);
}

// Eliminates unused array outputs from `remap_op` (or erases `remap_op`
// entirely and forwards its `control_inputs` to consumers of its
// `control_output` if no array outputs are used), pruning unreferenced input
// arrays and compacting `in_array_index` and `out_array_index` in kept
// `IfrtArrayMappingAttr` entries.
void EliminateUnusedRemapArraysOutputs(RemapArraysOp remap_op,
                                       CalleeSymbolManager& symbol_manager,
                                       mlir::RewriterBase& rewriter) {
  UsedOutputsSlice slice = CollectUsedOutputs(remap_op);
  if (slice.keep.none()) {
    if (CanForwardControlInputs(remap_op.getControlOutput(),
                                remap_op.getControlInputs())) {
      ForwardControlInputsToUsers(remap_op.getControlOutput(),
                                  remap_op.getControlInputs(), rewriter);
      symbol_manager.RecordOpErased(remap_op);
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

// Eliminates unused results and inputs of `call_op` and its resolved callee:
// erases `call_op` (plus its private callee if unused) when all outputs are
// unused and the callee is side-effect free, or clones a private copy of the
// callee (or enclosing module) when shared before erasing unused `call_op`
// outputs/inputs and remapping aliasing/donation/arg/result attributes. Returns
// `mlir::failure()` (having emitted an op error) if the callee cannot be
// resolved or prepared for mutation.
mlir::LogicalResult EliminateCallDce(CallOp call_op,
                                     CalleeSymbolManager& symbol_manager,
                                     mlir::RewriterBase& rewriter) {
  mlir::FailureOr<mlir::func::FuncOp> resolved_callee =
      symbol_manager.GetCallee(call_op);
  if (mlir::failed(resolved_callee)) {
    return mlir::failure();
  }
  mlir::func::FuncOp callee = *resolved_callee;

  UsedOutputsSlice out_slice = CollectUsedOutputs(call_op);

  // Erase dead `CallOp` (and its private callee if now unused) without cloning
  // when all outputs are unused, control inputs can be forwarded, and `callee`
  // is side-effect free.
  if (out_slice.kept_types.empty() &&
      CanForwardControlInputs(call_op.getControlOutput(),
                              call_op.getControlInputs()) &&
      IsSideEffectFreeCallee(callee)) {
    ForwardControlInputsToUsers(call_op.getControlOutput(),
                                call_op.getControlInputs(), rewriter);
    symbol_manager.EraseCallAndMaybeEraseCallee(call_op, rewriter);
    return mlir::success();
  }

  const bool has_unused_args =
      llvm::any_of(callee.getBody().front().getArguments(),
                   [](mlir::BlockArgument arg) { return arg.use_empty(); });
  // If no outputs are erased and there are no unused arguments, then the call
  // op and callee are already minimal.
  if (out_slice.erase.none() && !has_unused_args) {
    return mlir::success();
  }

  mlir::FailureOr<mlir::func::FuncOp> target_callee =
      symbol_manager.GetOrCloneCalleeForMutation({call_op});
  if (mlir::failed(target_callee)) {
    return mlir::failure();
  }
  callee = *target_callee;

  const unsigned num_inputs = call_op.getInputs().size();
  llvm::BitVector unused_arguments(num_inputs);
  llvm::SmallVector<int32_t> old_in_to_new_in(num_inputs, -1);
  llvm::SmallVector<mlir::Value> new_inputs;
  new_inputs.reserve(num_inputs);

  // Track symbol-use changes across result/argument erasure and region
  // simplification on `callee`.
  symbol_manager.RecordOpMutation(callee, [&]() {
    // 1. Eliminate unused `CallOp` results and corresponding callee return
    // values, then re-simplify the callee region so pure ops feeding only
    // erased return values are eliminated before checking block argument uses.
    if (out_slice.erase.any()) {
      for (mlir::Block& block : callee.getBody()) {
        if (auto return_op =
                llvm::dyn_cast<mlir::func::ReturnOp>(block.getTerminator());
            return_op != nullptr) {
          return_op->eraseOperands(out_slice.erase);
        }
      }
      CHECK(mlir::succeeded(callee.eraseResults(out_slice.erase)));
      (void)mlir::simplifyRegions(rewriter, callee.getBody());
    }

    // 2. Eliminate unused callee entry block arguments and `CallOp` inputs.
    mlir::Block& entry_block = callee.getBody().front();
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
          const int64_t old_out = aliasing_attr.getInt();
          CHECK_GE(old_out, 0);
          CHECK_LT(static_cast<size_t>(old_out), out_slice.old_to_new.size());
          if (out_slice.old_to_new[old_out] == -1) {
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
  });
  CHECK(out_slice.erase.any() || unused_arguments.any());

  // 3. Remap `donated_input_indices` and `io_aliases` and replace `call_op`.
  llvm::SmallVector<int32_t> new_donated_input_indices;
  for (int32_t old_in : call_op.getDonatedInputIndices()) {
    if (old_in_to_new_in[old_in] != -1) {
      new_donated_input_indices.push_back(old_in_to_new_in[old_in]);
    }
  }
  llvm::SmallVector<mlir::Attribute> new_io_aliases;
  for (const IoAlias& alias : call_op.getIoAliasPairs()) {
    if (old_in_to_new_in[alias.input_index] != -1 &&
        out_slice.old_to_new[alias.output_index] != -1) {
      new_io_aliases.push_back(rewriter.getDenseI32ArrayAttr(
          {old_in_to_new_in[alias.input_index],
           out_slice.old_to_new[alias.output_index]}));
    }
  }

  mlir::ArrayAttr new_arg_attrs =
      FilterDictArrayAttr(call_op.getArgAttrsAttr(), unused_arguments);
  mlir::ArrayAttr new_res_attrs =
      FilterDictArrayAttr(call_op.getResAttrsAttr(), out_slice.erase);

  rewriter.setInsertionPoint(call_op);
  CallOp new_call =
      CallOp::create(rewriter, call_op.getLoc(),
                     /*outputs=*/out_slice.kept_types,
                     /*control_output=*/call_op.getControlOutput().getType(),
                     /*inputs=*/new_inputs,
                     /*control_inputs=*/call_op.getControlInputs(),
                     /*arg_attrs=*/new_arg_attrs,
                     /*res_attrs=*/new_res_attrs,
                     /*callee=*/call_op.getCalleeAttr(),
                     /*devices=*/call_op.getDevicesAttr(),
                     /*io_aliases=*/rewriter.getArrayAttr(new_io_aliases),
                     /*donated_input_indices=*/
                     rewriter.getDenseI32ArrayAttr(new_donated_input_indices));
  new_call->setDiscardableAttrs(call_op->getDiscardableAttrDictionary());

  symbol_manager.RecordOpCreated(new_call);
  symbol_manager.RecordOpErased(call_op);
  ReplaceKeptOutputs(rewriter, call_op, new_call, out_slice.old_to_new);
  return mlir::success();
}

class IfrtDcePass : public impl::IfrtDcePassBase<IfrtDcePass> {
 public:
  using impl::IfrtDcePassBase<IfrtDcePass>::IfrtDcePassBase;

  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    mlir::IRRewriter rewriter(&getContext());
    mlir::SymbolTableCollection symbol_table;
    mlir::FailureOr<CalleeSymbolManager> symbol_manager =
        CalleeSymbolManager::Create(module_op, &symbol_table);
    if (mlir::failed(symbol_manager)) {
      signalPassFailure();
      return;
    }
    // Snapshot IFRT entry functions upfront because `EliminateCallDce` inserts
    // and erases private callee `func.func` ops in `module_op`, which would
    // invalidate a live `module_op.getOps<mlir::func::FuncOp>()` iterator.
    llvm::SmallVector<mlir::func::FuncOp> ifrt_funcs =
        GetIfrtFunctions(module_op);

    // Simplify dead pure ops and unreachable blocks inside every distinct
    // defined callee upfront (preserving callee signatures and semantics for
    // all callers) so dead symbol references inside unreachable blocks are
    // dropped before the reverse DCE walk checks callee exclusivity or erases
    // calls.
    llvm::DenseSet<mlir::func::FuncOp> simplified_callees;
    for (mlir::func::FuncOp func_op : ifrt_funcs) {
      for (CallOp call_op : func_op.getOps<CallOp>()) {
        mlir::FailureOr<mlir::func::FuncOp> callee =
            symbol_manager->GetCallee(call_op);
        if (mlir::failed(callee)) {
          signalPassFailure();
          return;
        }
        if (simplified_callees.insert(*callee).second) {
          symbol_manager->RecordOpMutation(*callee, [&]() {
            (void)mlir::simplifyRegions(rewriter, callee->getBody());
          });
        }
      }
    }

    for (mlir::func::FuncOp func_op : ifrt_funcs) {
      for (mlir::Operation* op = &func_op.getBody().front().back();
           op != nullptr;) {
        mlir::Operation* prev_op = op->getPrevNode();
        const mlir::LogicalResult op_result =
            llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
                .Case<CallOp>([&](CallOp call_op) {
                  return EliminateCallDce(call_op, *symbol_manager, rewriter);
                })
                .Case<IfrtPairwiseArrayOpInterface>(
                    [&](IfrtPairwiseArrayOpInterface pairwise_op) {
                      EliminateUnusedPairwiseArrayOpOutputs(
                          pairwise_op, *symbol_manager, rewriter);
                      return mlir::success();
                    })
                .Case<RemapArraysOp>([&](RemapArraysOp remap_op) {
                  EliminateUnusedRemapArraysOutputs(remap_op, *symbol_manager,
                                                    rewriter);
                  return mlir::success();
                })
                .Default(mlir::success());
        if (mlir::failed(op_result)) {
          signalPassFailure();
          return;
        }
        op = prev_op;
      }
    }

    symbol_manager->EraseUnusedTrackedRootSymbols();
  }
};

}  // namespace
}  // namespace ifrt
}  // namespace xla
