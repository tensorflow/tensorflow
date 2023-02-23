/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/op_utils.h"

#include <string>

#include "llvm/Support/raw_ostream.h"

namespace tensorflow {
namespace dtensor {

// OpHash prints the `op` into a string and performs hash value on the output
// string.
//
// The `print` includes the full representation of the `op`, e.g., target
// register, layout, shape, etc. This should be sufficient to uniquely
// identify the operation in most cases. This does not capture function scope
// (identical op in 2 separate functions).
uint64_t OpHash(mlir::Operation* op) {
  std::string output;
  llvm::raw_string_ostream output_stream(output);
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(1024);
  op->print(output_stream, flags);
  return llvm::hash_value(output);
}

// Returns FuncOp if `op` is a callable.
absl::optional<mlir::func::FuncOp> MaybeFindFunction(mlir::Operation* op) {
  auto call_op = llvm::dyn_cast<mlir::CallOpInterface>(op);
  if (!call_op) return absl::nullopt;

  mlir::CallInterfaceCallable callable = call_op.getCallableForCallee();
  mlir::SymbolRefAttr sym = callable.dyn_cast<mlir::SymbolRefAttr>();
  if (!sym) return absl::nullopt;

  mlir::func::FuncOp func = llvm::dyn_cast<mlir::func::FuncOp>(
      mlir::SymbolTable::lookupNearestSymbolFrom(op, sym));
  if (!func) return absl::nullopt;

  return func;
}

void RemoveDTensorLayoutOps(mlir::ModuleOp module,
                            bool remove_xla_spmd_layouts) {
  llvm::SmallVector<mlir::TF::DTensorLayout, 4> layout_ops;
  module.walk([&](mlir::TF::DTensorLayout layout) {
    // Remove layout ops only for layouts running on DTensor SPMD.
    // Layout ops will be preserved for XLA SPMD to annotate sharding
    // later down the DTensor stack.
    if (remove_xla_spmd_layouts || !layout.getLayout().mesh().use_xla_spmd()) {
      layout_ops.emplace_back(layout);
    }
  });

  for (auto layout_op : layout_ops) {
    layout_op.getOutput().replaceAllUsesWith(layout_op.getInput());
    layout_op.erase();
  }
}

mlir::LogicalResult ReplaceAuxiliaryDTensorLayoutOpsWithIdentity(
    mlir::ModuleOp module) {
  llvm::SmallVector<mlir::TF::DTensorLayout, 4> layout_ops;
  module.walk([&](mlir::TF::DTensorLayout op) { layout_ops.emplace_back(op); });

  llvm::DenseSet<mlir::TF::DTensorLayout> deleted_layout_ops;

  for (auto layout_op : llvm::reverse(layout_ops)) {
    if (deleted_layout_ops.contains(layout_op)) {
      continue;
    }
    while (auto input_layout_op =
               llvm::dyn_cast_or_null<mlir::TF::DTensorLayout>(
                   layout_op.getInput().getDefiningOp())) {
      // Check that layout of input DTensorLayout op is equivalent to
      // the layout of its connected DTensorLayout op.
      if (layout_op.getLayout() != input_layout_op.getLayout()) {
        return layout_op.emitOpError(
            "Found inconsistent layout. This should never happen.");
      }

      // Replace DTensorLayout op with identity op.
      mlir::OpBuilder builder(input_layout_op);
      auto new_identity = builder.create<mlir::TF::IdentityOp>(
          input_layout_op->getLoc(), input_layout_op.getType(),
          input_layout_op.getInput());
      input_layout_op.getOutput().replaceAllUsesWith(new_identity.getOutput());
      input_layout_op.erase();

      deleted_layout_ops.insert(input_layout_op);
    }
  }

  return mlir::success();
}
}  // namespace dtensor
}  // namespace tensorflow
