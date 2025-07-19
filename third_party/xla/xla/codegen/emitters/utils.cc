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

#include "xla/codegen/emitters/utils.h"

#include <iterator>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::emitters {

using mlir::DenseElementsAttr;
using mlir::ShapedType;

DenseElementsAttr GetZeroDenseElementsAttr(ShapedType shaped_type) {
  auto elem_type = shaped_type.getElementType();
  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(elem_type)) {
    mlir::SmallVector<llvm::APFloat, 4> values(
        shaped_type.getNumElements(),
        mlir::APFloat::getZero(float_type.getFloatSemantics()));
    return DenseElementsAttr::get(shaped_type, values);
  }
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(elem_type)) {
    mlir::SmallVector<llvm::APInt, 4> values(
        shaped_type.getNumElements(),
        mlir::APInt::getZero(int_type.getIntOrFloatBitWidth()));
    return DenseElementsAttr::get(shaped_type, values);
  }
  llvm_unreachable("Unsupported element type");
}

absl::flat_hash_map<const HloInstruction*, mlir::ValueRange> EmitEpilogue(
    int epilogue_index, const emitters::PartitionedComputations& computations,
    mlir::func::FuncOp entry_fn,
    const absl::flat_hash_map<const HloInstruction*,
                              llvm::SmallVector<mlir::Value>>& injected,
    mlir::ValueRange output_indices, mlir::ImplicitLocOpBuilder& builder) {
  const auto& epilogue = computations.epilogues().at(epilogue_index);
  if (epilogue.roots.empty()) {
    return {};
  }
  auto epilogue_fn = mlir::cast<mlir::func::FuncOp>(
      entry_fn->getParentOfType<mlir::ModuleOp>().lookupSymbol(epilogue.name));
  llvm::SmallVector<mlir::Value> operands =
      mlir::ValueRange(entry_fn.getArguments().take_front(
          computations.fusion()->num_parameters()));
  absl::c_copy(output_indices, std::back_inserter(operands));
  int injected_offset = operands.size();
  operands.resize(injected_offset + epilogue.num_injected_values);
  for (auto [injected_instruction, start] : epilogue.injected_value_starts) {
    absl::c_copy(injected.at(injected_instruction),
                 operands.begin() + injected_offset + start);
  }

  mlir::ValueRange results =
      builder.create<PureCallOp>(epilogue_fn, operands).getResults();
  absl::flat_hash_map<const HloInstruction*, mlir::ValueRange> results_per_root;
  for (auto* root : epilogue.roots) {
    int arity =
        root->shape().IsTuple() ? root->shape().tuple_shapes().size() : 1;
    results_per_root[root] = results.take_front(arity);
    results = results.drop_front(arity);
  }
  CHECK_EQ(results.size(), 0);
  return results_per_root;
}

}  // namespace xla::emitters
