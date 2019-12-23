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

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/protobuf/tpu/dynamic_padding.pb.h"

namespace mlir {
namespace TFTPU {

constexpr char kPaddingMapAttr[] = "padding_map";

// This pass remaps and assigns padding maps to an encapsulated function's
// arguments from a `tf_device.launch_func` `padding_map` attribute. Remapping
// is from replicated input index to encapsulated function's operand index
// (user).

namespace {
struct TPUDynamicPaddingMapper : public ModulePass<TPUDynamicPaddingMapper> {
  void runOnModule() override;
};

// Creates a mapping from replicated input index (in `tf_device.replicate` op)
// to `tf_device.launch_func` operand index.
llvm::SmallDenseMap<int32_t, int32_t> GetRemappedReplicatedInputIndices(
    tf_device::LaunchFuncOp launch_func, tf_device::ReplicateOp replicate) {
  Block* replicate_block = &replicate.GetBody();

  llvm::SmallDenseMap<int32_t, int32_t> remapped_indices;
  for (auto operand_and_idx : llvm::enumerate(launch_func.getOperands()))
    if (auto block_arg = operand_and_idx.value()->dyn_cast<BlockArgument>())
      if (block_arg->getOwner() == replicate_block)
        remapped_indices[block_arg->getArgNumber()] = operand_and_idx.index();

  return remapped_indices;
}

// Extracts `padding_map` from `tf_device.launch_func` and remaps the associated
// replicated input indices to the encapsulated function operand indices. An
// error will be returned if an index is not found or parsing failed.
LogicalResult GetRemappedPaddings(
    tf_device::LaunchFuncOp launch_func, int num_replicated_args,
    const llvm::SmallDenseMap<int32_t, int32_t>& remapped_indices,
    llvm::SmallVectorImpl<tensorflow::tpu::PaddingMap>* remapped_paddings) {
  auto bad_index_msg = [num_replicated_args](int32_t index,
                                             llvm::StringRef arg_type,
                                             int32_t arg_index) {
    return llvm::formatv(
               "bad '{0}' attribute at index {1}, {2} must be in [0, {3}), got "
               "{4}",
               kPaddingMapAttr, index, arg_type, num_replicated_args, arg_index)
        .str();
  };

  Attribute padding_map_attr = launch_func.getAttr(kPaddingMapAttr);
  if (!padding_map_attr) return success();

  auto padding_map = padding_map_attr.dyn_cast<ArrayAttr>();
  if (!padding_map)
    return launch_func.emitOpError()
           << "requires '" << kPaddingMapAttr << "' array attribute";

  for (auto padding_attr_and_idx : llvm::enumerate(padding_map)) {
    int idx = padding_attr_and_idx.index();
    auto& padding_attr = padding_attr_and_idx.value();
    auto padding = padding_attr.dyn_cast<StringAttr>();
    if (!padding)
      return launch_func.emitOpError(
          llvm::formatv("bad '{0}' attribute at index {1}, not a string",
                        kPaddingMapAttr, padding_attr_and_idx.index()));

    tensorflow::tpu::PaddingMap padding_proto;
    if (!padding_proto.ParseFromString(padding.getValue().str()))
      return launch_func.emitOpError(llvm::formatv(
          "bad '{0}' attribute at index {1}, failed to parse '{2}' as "
          "tensorflow::tpu::PaddingMap",
          kPaddingMapAttr, idx, padding.getValue()));

    const int32_t arg_index = padding_proto.arg_index();
    if (arg_index >= num_replicated_args || arg_index < 0)
      return launch_func.emitOpError()
             << bad_index_msg(idx, "arg_index", arg_index);

    const int32_t padding_arg_index = padding_proto.padding_arg_index();
    if (padding_arg_index >= num_replicated_args || padding_arg_index < 0)
      return launch_func.emitOpError()
             << bad_index_msg(idx, "padding_arg_index", padding_arg_index);

    auto arg_index_it = remapped_indices.find(arg_index);
    // Skip unused arguments.
    if (arg_index_it == remapped_indices.end()) continue;

    auto padding_arg_index_it = remapped_indices.find(padding_arg_index);
    if (padding_arg_index_it == remapped_indices.end())
      return launch_func.emitOpError(llvm::formatv(
          "bad '{0}' attribute at index {1}, unused padding_arg_index {2}",
          kPaddingMapAttr, idx, padding_arg_index));

    padding_proto.set_arg_index(arg_index_it->second);
    padding_proto.set_padding_arg_index(padding_arg_index_it->getSecond());
    remapped_paddings->push_back(std::move(padding_proto));
  }

  return success();
}

// Inserts padding maps for relevant arguments as argument attributes on the
// encapsulated function. The padding maps will be in the form of:
//   %arg0 : type {xla_hlo.padding_map = {shape_indices = [...],
//                                        padding_arg_indices = [...]}}
void AnnotateFunctionArgumentsWithPaddings(
    FuncOp func,
    llvm::ArrayRef<tensorflow::tpu::PaddingMap> remapped_paddings) {
  // Group paddings by arg index.
  llvm::SmallDenseMap<int32_t, std::pair<llvm::SmallVector<int32_t, 4>,
                                         llvm::SmallVector<int32_t, 4>>>
      paddings;
  for (const auto& padding : remapped_paddings) {
    auto& it = paddings[padding.arg_index()];
    it.first.push_back(padding.shape_index());
    it.second.push_back(padding.padding_arg_index());
  }

  Builder builder(func.getContext());
  for (const auto& padding : paddings) {
    auto shape_indices = builder.getNamedAttr(
        "shape_indices", builder.getI32ArrayAttr(padding.getSecond().first));
    auto padding_arg_indices = builder.getNamedAttr(
        "padding_arg_indices",
        builder.getI32ArrayAttr(padding.getSecond().second));
    func.setArgAttr(
        padding.getFirst(), "xla_hlo.padding_map",
        builder.getDictionaryAttr({shape_indices, padding_arg_indices}));
  }
}

LogicalResult RemapAndAssignPaddingMaps(tf_device::LaunchFuncOp launch_func,
                                        SymbolTable* symbol_table) {
  auto replicate =
      llvm::dyn_cast_or_null<tf_device::ReplicateOp>(launch_func.getParentOp());
  // LaunchFunc is not replicated, there will be no padding.
  if (!replicate) return success();
  const int num_replicated_args = replicate.GetBody().getNumArguments();

  auto func = symbol_table->lookup<FuncOp>(launch_func.func());
  if (!func) return success();

  llvm::SmallDenseMap<int32_t, int32_t> remapped_indices =
      GetRemappedReplicatedInputIndices(launch_func, replicate);

  llvm::SmallVector<tensorflow::tpu::PaddingMap, 4> remapped_paddings;
  if (failed(GetRemappedPaddings(launch_func, num_replicated_args,
                                 remapped_indices, &remapped_paddings)))
    return failure();

  AnnotateFunctionArgumentsWithPaddings(func, remapped_paddings);

  return success();
}

void TPUDynamicPaddingMapper::runOnModule() {
  ModuleOp module = getModule();
  SymbolTable symbol_table(module);
  module.walk([&](tf_device::LaunchFuncOp launch_func) {
    RemapAndAssignPaddingMaps(launch_func, &symbol_table);
  });
}
}  // anonymous namespace

std::unique_ptr<OpPassBase<ModuleOp>> CreateTPUDynamicPaddingMapperPass() {
  return std::make_unique<TPUDynamicPaddingMapper>();
}

static PassRegistration<TPUDynamicPaddingMapper> pass(
    "tf-tpu-dynamic-padding",
    "Remaps padding map from replicated inputs to argument ordering on "
    "encapsulated function");

}  // namespace TFTPU
}  // namespace mlir
