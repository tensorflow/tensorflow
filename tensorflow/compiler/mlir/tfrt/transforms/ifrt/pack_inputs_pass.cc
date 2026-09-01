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

#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/pack_inputs_pass.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace tensorflow {
namespace ifrt_serving {

PackInputsPass::PackInputsPass(llvm::ArrayRef<SliceInfo> slices)
    : slices_(slices.begin(), slices.end()) {}

// TODO(b/445201291): This pass currently has a caveat of losing the original
// user inputs' layout. We should look into this if it turns out to be a
// problem.
void PackInputsPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_func || main_func.isPrivate() || main_func.isExternal()) {
    module.emitError("Expected a public function 'main'.");
    signalPassFailure();
    return;
  }

  // If command line option is provided, use it to populate slices_
  if (!slices_flat_list_.empty()) {
    if (slices_flat_list_.size() % 3 != 0) {
      module.emitError(
          "The 'slices' option must be a flat list of integers "
          "with a length multiple of 3.");
      signalPassFailure();
      return;
    }
    slices_.clear();
    for (size_t i = 0; i < slices_flat_list_.size(); i += 3) {
      slices_.push_back({static_cast<unsigned>(slices_flat_list_[i]),
                         slices_flat_list_[i + 1], slices_flat_list_[i + 2]});
    }
  }

  if (slices_.empty()) {
    return;
  }

  llvm::SmallVector<unsigned> small_arg_indices;
  int64_t total_small_size = 0;

  for (const auto& slice : slices_) {
    if (slice.arg_index >= main_func.getNumArguments()) {
      module->emitError(absl::StrCat("Slice arg_index ", slice.arg_index,
                                     " is out of range."));
      return signalPassFailure();
    }

    if (llvm::is_contained(small_arg_indices, slice.arg_index)) {
      module->emitError(
          absl::StrCat("Duplicate slice arg_index: ", slice.arg_index));
      return signalPassFailure();
    }

    mlir::BlockArgument arg = main_func.getArgument(slice.arg_index);
    auto tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(arg.getType());
    if (!tensor_type) {
      module->emitError(absl::StrCat("Argument ", slice.arg_index,
                                     " is not a ranked tensor type."));
      return signalPassFailure();
    }
    if (!tensor_type.hasStaticShape()) {
      module->emitError(absl::StrCat("Argument ", slice.arg_index,
                                     " does not have static shape."));
      return signalPassFailure();
    }

    int64_t bitwidth = tensor_type.getElementType().getIntOrFloatBitWidth();
    if (bitwidth < 8 || bitwidth % 8 != 0) {
      module->emitError(
          absl::StrCat("Argument ", slice.arg_index,
                       " has invalid element bit width: ", bitwidth));
      return signalPassFailure();
    }

    int64_t expected_size = tensor_type.getNumElements() * (bitwidth / 8);
    if (slice.size != expected_size) {
      module->emitError(absl::StrCat(
          "Slice size ", slice.size, " for argument ", slice.arg_index,
          " does not match expected byte size ", expected_size));
      return signalPassFailure();
    }

    small_arg_indices.push_back(slice.arg_index);
    total_small_size = std::max(total_small_size, slice.start + slice.size);
  }

  // Check for overlapping slices
  for (size_t i = 0; i < slices_.size(); ++i) {
    for (size_t j = i + 1; j < slices_.size(); ++j) {
      int64_t start_i = slices_[i].start;
      int64_t end_i = start_i + slices_[i].size;
      int64_t start_j = slices_[j].start;
      int64_t end_j = start_j + slices_[j].size;
      if (std::max(start_i, start_j) < std::min(end_i, end_j)) {
        module->emitError(absl::StrCat("Slices for argument ",
                                       slices_[i].arg_index, " and ",
                                       slices_[j].arg_index, " overlap."));
        return signalPassFailure();
      }
    }
  }

  mlir::OpBuilder builder(&getContext());
  mlir::RankedTensorType combined_arg_type =
      mlir::RankedTensorType::get({total_small_size}, builder.getI8Type());

  mlir::Block& entry_block = main_func.front();
  mlir::BlockArgument combined_arg =
      entry_block.addArgument(combined_arg_type, main_func.getLoc());

  builder.setInsertionPointToStart(&entry_block);
  for (const auto& slice : slices_) {
    mlir::BlockArgument arg = entry_block.getArgument(slice.arg_index);
    llvm::SmallVector<int64_t> start_indices = {slice.start};
    llvm::SmallVector<int64_t> limit_indices = {slice.start + slice.size};
    llvm::SmallVector<int64_t> strides = {1};
    auto slice_op = builder.create<mlir::stablehlo::SliceOp>(
        arg.getLoc(), combined_arg, builder.getDenseI64ArrayAttr(start_indices),
        builder.getDenseI64ArrayAttr(limit_indices),
        builder.getDenseI64ArrayAttr(strides));

    auto tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(arg.getType());
    int64_t targetEltBitWidth =
        tensor_type.getElementType().getIntOrFloatBitWidth();

    // If the original type is one byte, we can just do a reshape. If it is
    // larger than one byte, we need to do a reshape and a bitcast to convert
    // to the original type. For example, if the original type is F32 of shape
    // (x, y), we need to reshape to {x, y, 4} as F32 is 4 bytes, and then
    // bitcast to F32 with shape (x, y).
    if (targetEltBitWidth == 8) {
      auto reshape_op = builder.create<mlir::stablehlo::ReshapeOp>(
          arg.getLoc(), arg.getType(), slice_op.getResult());
      arg.replaceAllUsesWith(reshape_op.getResult());
    } else {
      llvm::SmallVector<int64_t> intermediate_shape(
          tensor_type.getShape().begin(), tensor_type.getShape().end());
      intermediate_shape.push_back(targetEltBitWidth / 8);
      auto intermediate_type =
          mlir::RankedTensorType::get(intermediate_shape, builder.getI8Type());

      auto reshape_op = builder.create<mlir::stablehlo::ReshapeOp>(
          arg.getLoc(), intermediate_type, slice_op.getResult());

      auto bitcast_op = builder.create<mlir::stablehlo::BitcastConvertOp>(
          arg.getLoc(), arg.getType(), reshape_op.getResult());

      arg.replaceAllUsesWith(bitcast_op.getResult());
    }
  }

  // Erase arguments
  llvm::BitVector small_args_to_erase_bv(entry_block.getNumArguments());
  for (unsigned index : small_arg_indices) {
    small_args_to_erase_bv.set(index);
  }
  entry_block.eraseArguments(small_args_to_erase_bv);

  // Set new function type and attributes
  mlir::FunctionType old_func_type = main_func.getFunctionType();
  llvm::SmallVector<mlir::Type> new_input_types;
  llvm::SmallVector<mlir::DictionaryAttr> new_arg_attrs;
  for (unsigned i = 0; i < old_func_type.getNumInputs(); ++i) {
    if (!llvm::is_contained(small_arg_indices, i)) {
      new_input_types.push_back(old_func_type.getInput(i));
      new_arg_attrs.push_back(main_func.getArgAttrDict(i));
    }
  }
  new_input_types.push_back(combined_arg_type);
  new_arg_attrs.push_back(nullptr);

  auto new_func_type = mlir::FunctionType::get(&getContext(), new_input_types,
                                               old_func_type.getResults());
  main_func.setFunctionType(new_func_type);
  main_func.setAllArgAttrs(new_arg_attrs);
}

void PackInputsPass::getDependentDialects(
    mlir::DialectRegistry& registry) const {
  registry.insert<mlir::stablehlo::StablehloDialect>();
}

mlir::StringRef PackInputsPass::getArgument() const { return "pack-inputs"; }

mlir::StringRef PackInputsPass::getDescription() const {
  return "Pack specified tensor inputs of main function into a single i8 "
         "tensor.";
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreatePackInputsPass(
    llvm::ArrayRef<SliceInfo> slices) {
  return std::make_unique<PackInputsPass>(slices);
}

}  // namespace ifrt_serving
}  // namespace tensorflow
