/* Copyright 2024 The OpenXLA Authors.

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
#include <cassert>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_MERGEPOINTERSTOSAMESLICEPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

static constexpr absl::string_view kXlaEntryAttr = "xla.entry";

class MergePointersToSameSlicePass
    : public impl::MergePointersToSameSlicePassBase<
          MergePointersToSameSlicePass> {
 public:
  void runOnOperation() override;
};

struct PackedArgs {
  llvm::BitVector args_to_erase;
  // replacement_args[i] == i iff !args_to_erase[i].
  llvm::SmallVector<int> replacement_args;

  PackedArgs() = default;
  explicit PackedArgs(mlir::func::FuncOp func) {
    absl::flat_hash_map<int, std::optional<int>> slice_to_operand;
    args_to_erase.resize(func.getNumArguments());
    replacement_args.reserve(func.getNumArguments());
    for (int i = 0; i < func.getNumArguments(); ++i) {
      replacement_args.push_back(i);
    }

    // Do not erase arguments for the entry function, because that will change
    // kernel launch interface.
    bool is_entry_func = func->hasAttr(kXlaEntryAttr);

    for (auto [idx, operand] : llvm::enumerate(func.getArguments())) {
      auto slice_index = func.getArgAttr(idx, "xla.slice_index");
      if (!slice_index) {
        continue;
      }

      auto& target_index = slice_to_operand[static_cast<int>(
          mlir::cast<mlir::IntegerAttr>(slice_index).getInt())];
      if (target_index) {
        replacement_args[idx] = *target_index;
        args_to_erase[idx] = !is_entry_func;
      } else {
        target_index = idx;
      }
    }
  }

  void Pack(mlir::func::FuncOp op) {
    for (auto [idx, arg] : llvm::enumerate(op.getArguments())) {
      if (replacement_args[idx] != idx) {
        arg.replaceAllUsesWith(op.getArgument(replacement_args[idx]));
      }
    }

    [[maybe_unused]] auto res = op.eraseArguments(args_to_erase);
    assert(llvm::succeeded(res));

    for (int i = 0; i < op.getNumArguments(); ++i) {
      if (op.getArgAttr(i, "xla.slice_index")) {
        op.removeArgAttr(i, "xla.slice_index");
        op.setArgAttr(i, mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                      mlir::UnitAttr::get(op->getContext()));
      }
    }
  }

  void Pack(mlir::func::CallOp op) { op->eraseOperands(args_to_erase); }
};

void MergePointersToSameSlicePass::runOnOperation() {
  absl::flat_hash_map<std::string, PackedArgs> args_to_pack;
  getOperation()->walk([&](mlir::func::FuncOp func) {
    args_to_pack[func.getName()] = PackedArgs(func);
  });
  getOperation()->walk([&](mlir::func::CallOp call) {
    args_to_pack[call.getCallee()].Pack(call);
  });
  getOperation()->walk([&](mlir::func::FuncOp func) {
    args_to_pack[func.getName()].Pack(func);
  });
}

}  // namespace

std::unique_ptr<mlir::Pass> CreateMergePointersToSameSlicePass() {
  return std::make_unique<MergePointersToSameSlicePass>();
}

}  // namespace emitters
}  // namespace xla
