/* Copyright 2025 The OpenXLA Authors.

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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"

namespace xla::emitters {

#define GEN_PASS_DECL_PROPAGATEALIASSCOPESPASS
#define GEN_PASS_DEF_PROPAGATEALIASSCOPESPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

auto kSliceIndexAttrName = "xla.slice_index";
auto kInvariantAttrName = "xla.invariant";

class PropagateAliasScopesPass final
    : public impl::PropagateAliasScopesPassBase<PropagateAliasScopesPass> {
 public:
  using PropagateAliasScopesPassBase::PropagateAliasScopesPassBase;

  void runOnOperation() override;

 private:
  // Main callback for the walking the ops withing the function.
  mlir::WalkResult WalkCallback(mlir::Operation* op);
  // Propagate the slice index to the arguments of the function.
  mlir::WalkResult WalkCall(mlir::CallOpInterface call_op);
  // Propagate the slice index to the iter-args of the for loop.
  mlir::WalkResult WalkFor(mlir::scf::ForOp for_op);
  // Add the noalias scope to the extract op.
  mlir::WalkResult WalkExtract(mlir::tensor::ExtractOp extract_op);
  // Add the alias & noalias scope to the insert op.
  mlir::WalkResult WalkInsert(mlir::tensor::InsertOp insert_op);

  // Initialize mapping from value to slice index and slice index to alias
  // scope.
  void InitializeBookeeping(mlir::func::FuncOp func_op);
  void InitializeAliasScopeBookeeping(mlir::func::FuncOp func_op);
  void InitializeNoAliasScopeBookeeping(mlir::MLIRContext* context);

 private:
  // Value to the slice index of the function argument.
  llvm::DenseMap<mlir::Value, int64_t> value_to_index_;
  // Slice index to the alias scope attribute.
  absl::flat_hash_map<int64_t, mlir::LLVM::AliasScopeAttr> index_to_alias_;
  // Slice index to the set of no alias scope attributes.
  absl::flat_hash_map<int64_t, mlir::ArrayAttr> index_to_no_alias_;
};

static void SetAliasScopeMetadata(
    mlir::Operation& op, const mlir::LLVM::AliasScopeAttr& alias_scope) {
  op.setAttr(mlir::LLVM::LLVMDialect::getAliasScopesAttrName(),
             mlir::ArrayAttr::get(op.getContext(), alias_scope));
}

static void SetNoAliasScopeMetadata(mlir::Operation& op,
                                    mlir::ArrayAttr alias_scopes) {
  if (alias_scopes.empty()) {
    return;
  }
  op.setAttr(mlir::LLVM::LLVMDialect::getNoAliasAttrName(), alias_scopes);
}

void PropagateAliasScopesPass::runOnOperation() {
  mlir::ModuleOp module_op = getOperation();

  mlir::func::FuncOp entry;
  for (auto func : getOperation().getOps<mlir::func::FuncOp>()) {
    if (func->getAttr("xla.entry")) {
      entry = func;
      break;
    }
  }

  if (!entry) {
    getOperation()->emitOpError("No entry function found.");
    signalPassFailure();
    return;
  }

  InitializeBookeeping(entry);

  module_op->walk<mlir::WalkOrder::PreOrder>(
      [this](mlir::Operation* op) { return WalkCallback(op); });
}

mlir::WalkResult PropagateAliasScopesPass::WalkCallback(mlir::Operation* op) {
  if (auto call_op = mlir::dyn_cast_or_null<mlir::CallOpInterface>(op)) {
    return WalkCall(call_op);
  }

  if (auto for_op = mlir::dyn_cast_or_null<mlir::scf::ForOp>(op)) {
    return WalkFor(for_op);
  }

  if (auto insert_op = mlir::dyn_cast_or_null<mlir::tensor::InsertOp>(op)) {
    return WalkInsert(insert_op);
  }

  if (auto extract_op = mlir::dyn_cast_or_null<mlir::tensor::ExtractOp>(op)) {
    return WalkExtract(extract_op);
  }

  return mlir::WalkResult::advance();
}

mlir::WalkResult PropagateAliasScopesPass::WalkCall(
    mlir::CallOpInterface call_op) {
  mlir::func::FuncOp callee =
      mlir::dyn_cast<mlir::func::FuncOp>(call_op.resolveCallable());
  if (!callee) {
    // Could be a call to an external function.
    return mlir::WalkResult::advance();
  }

  // Forward the slice index to the arguments of the callee.
  for (auto [arg, operand] :
       llvm::zip(callee.getArguments(), call_op.getArgOperands())) {
    auto slice_index_itr = value_to_index_.find(operand);
    if (slice_index_itr != value_to_index_.end()) {
      value_to_index_.insert({arg, slice_index_itr->second});
    }
  }

  return mlir::WalkResult::advance();
}

mlir::WalkResult PropagateAliasScopesPass::WalkFor(mlir::scf::ForOp for_op) {
  // Forward the slice index to the iter-args of the for loop.
  for (auto [arg, operand] :
       llvm::zip(for_op.getRegionIterArgs(), for_op.getInitArgs())) {
    auto slice_index_itr = value_to_index_.find(operand);
    if (slice_index_itr != value_to_index_.end()) {
      value_to_index_.insert({arg, slice_index_itr->second});
    }
  }

  return mlir::WalkResult::advance();
}

mlir::WalkResult PropagateAliasScopesPass::WalkExtract(
    mlir::tensor::ExtractOp extract_op) {
  auto index_itr = value_to_index_.find(extract_op.getTensor());
  if (index_itr == value_to_index_.end()) {
    return mlir::WalkResult::advance();
  }

  int64_t slice_index = index_itr->second;

  if (auto no_alias_itr = index_to_no_alias_.find(slice_index);
      no_alias_itr != index_to_no_alias_.end()) {
    SetNoAliasScopeMetadata(*extract_op, no_alias_itr->second);
  }

  return mlir::WalkResult::advance();
}

mlir::WalkResult PropagateAliasScopesPass::WalkInsert(
    mlir::tensor::InsertOp insert_op) {
  auto index_itr = value_to_index_.find(insert_op.getDest());
  if (index_itr == value_to_index_.end()) {
    return mlir::WalkResult::advance();
  }

  int64_t slice_index = index_itr->second;

  if (const auto alias_itr = index_to_alias_.find(slice_index);
      alias_itr != index_to_alias_.end()) {
    SetAliasScopeMetadata(*insert_op, alias_itr->second);
  }

  if (auto no_alias_itr = index_to_no_alias_.find(slice_index);
      no_alias_itr != index_to_no_alias_.end()) {
    SetNoAliasScopeMetadata(*insert_op, no_alias_itr->second);
  }

  return mlir::WalkResult::advance();
}

void PropagateAliasScopesPass::InitializeBookeeping(
    mlir::func::FuncOp func_op) {
  InitializeAliasScopeBookeeping(func_op);
  InitializeNoAliasScopeBookeeping(func_op.getContext());
}

void PropagateAliasScopesPass::InitializeAliasScopeBookeeping(
    mlir::func::FuncOp func_op) {
  value_to_index_.clear();
  index_to_alias_.clear();

  auto domain = mlir::LLVM::AliasScopeDomainAttr::get(func_op.getContext());
  for (mlir::BlockArgument arg : func_op.getArguments()) {
    auto slice_index_attr = func_op.getArgAttrOfType<mlir::IntegerAttr>(
        arg.getArgNumber(), kSliceIndexAttrName);
    if (!slice_index_attr) {
      continue;
    }

    value_to_index_.insert({arg, slice_index_attr.getInt()});

    // We only need to set the alias scope for arguments that are written to.
    if (func_op.getArgAttr(arg.getArgNumber(), kInvariantAttrName)) {
      continue;
    }

    int64_t slice_index = slice_index_attr.getInt();
    auto scope = mlir::LLVM::AliasScopeAttr::get(
        domain, mlir::StringAttr::get(
                    func_op.getContext(),
                    absl::StrCat(kSliceIndexAttrName, "=", slice_index)));
    index_to_alias_.insert({slice_index, scope});
  }
}

void PropagateAliasScopesPass::InitializeNoAliasScopeBookeeping(
    mlir::MLIRContext* context) {
  index_to_no_alias_.clear();

  for (const auto& [value, slice_index] : value_to_index_) {
    std::vector<mlir::Attribute> no_alias;
    for (const auto& [inner_slice_index, no_alias_scope] : index_to_alias_) {
      if (inner_slice_index != slice_index) {
        no_alias.push_back(no_alias_scope);
      }
    }
    index_to_no_alias_.insert(
        {slice_index, mlir::ArrayAttr::get(context, no_alias)});
  }
}

}  // namespace xla::emitters
