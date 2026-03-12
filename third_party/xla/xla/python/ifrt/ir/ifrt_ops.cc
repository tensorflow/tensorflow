/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/ifrt_ops.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"

// Generated definitions.
#define GET_OP_CLASSES
#include "xla/python/ifrt/ir/ifrt_ops.cc.inc"

namespace xla {
namespace ifrt {

namespace {

mlir::FailureOr<mlir::RankedTensorType> GetGlobalShape(mlir::Type type) {
  if (auto ranked_tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    return ranked_tensor;
  }
  if (auto array = mlir::dyn_cast<IfrtArrayType>(type)) {
    return array.getShape();
  }
  return mlir::failure();
}

mlir::FailureOr<mlir::RankedTensorType> GetGlobalShape(mlir::Value value) {
  return GetGlobalShape(value.getType());
}

mlir::FailureOr<mlir::RankedTensorType> GetGlobalShapeFromLocal(
    mlir::Type type, IfrtShardingAttrInterface sharding_attr) {
  if (auto local_ranked_tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    auto global_shape =
        sharding_attr.GlobalShapeFromLocalShape(local_ranked_tensor.getShape());
    if (global_shape.ok()) {
      return mlir::RankedTensorType::get(global_shape.value(),
                                         local_ranked_tensor.getElementType());
    }
    return mlir::failure();
  }
  // IFRT arrays cannot be in the local view.
  return mlir::failure();
}

template <typename T, typename U>
mlir::LogicalResult VerifySameGlobalShape(mlir::Operation* op,
                                          llvm::StringRef lhs_mnemonic, T lhs,
                                          llvm::StringRef rhs_mnemonic, U rhs) {
  mlir::FailureOr<mlir::RankedTensorType> lhs_shape = GetGlobalShape(lhs);
  if (mlir::failed(lhs_shape)) {
    return op->emitOpError()
           << "fails to get global shape from " << lhs_mnemonic << ": " << lhs;
  }
  mlir::FailureOr<mlir::RankedTensorType> rhs_shape = GetGlobalShape(rhs);
  if (mlir::failed(rhs_shape)) {
    return op->emitOpError()
           << "fails to get global shape from " << rhs_mnemonic << ": " << rhs;
  }
  if (*lhs_shape != *rhs_shape) {
    return op->emitOpError()
           << "requires the same global shape. " << lhs_mnemonic << " "
           << *lhs_shape << " vs " << rhs_mnemonic << " " << *rhs_shape;
  }
  return mlir::success();
}

// Verifies that the global shape of a call op argument/result is the same
// as the global shape of corresponding argument/result of the function in
// local view.
mlir::LogicalResult VerifyGlobalLocalShapesEquivalent(
    mlir::Operation* op, llvm::StringRef call_mnemonic, mlir::Value call_value,
    llvm::StringRef callee_mnemonic, mlir::Type callee_type) {
  // The call values are in the global view.
  mlir::FailureOr<mlir::RankedTensorType> call_shape =
      GetGlobalShape(call_value);
  if (mlir::failed(call_shape)) {
    return op->emitOpError() << "fails to get global shape from "
                             << call_mnemonic << ": " << call_value;
  }
  // The types of the CallOp func signature must be IfrtArrayType.
  auto array = mlir::dyn_cast<IfrtArrayType>(call_value.getType());
  if (array == nullptr) {
    return mlir::failure();
  }
  // Convert from local shape to global shape using the sharding provided
  // by the CallOp func signature.
  mlir::FailureOr<mlir::RankedTensorType> callee_shape =
      GetGlobalShapeFromLocal(callee_type, array.getShardingAttr());
  if (mlir::failed(callee_shape)) {
    return op->emitOpError() << "fails to get global shape from "
                             << callee_mnemonic << ": " << callee_type;
  }
  if (*call_shape != *callee_shape) {
    return op->emitOpError()
           << "requires the same global shape. " << call_mnemonic << " "
           << *call_shape << " vs " << callee_mnemonic << " " << *callee_shape;
  }
  return mlir::success();
}

// Verifies that each of `inputs` and `outputs` is placed on a subset of
// `devices`.
mlir::LogicalResult VerifyDevicePlacement(
    mlir::Operation* op, llvm::ArrayRef<int> devices,
    llvm::ArrayRef<IfrtArrayType> inputs,
    llvm::ArrayRef<IfrtArrayType> outputs) {
  llvm::DenseSet<int> device_set;
  device_set.insert(devices.begin(), devices.end());

  for (const IfrtArrayType input : inputs) {
    for (const int input_device : input.getDevices()) {
      if (!device_set.count(input_device)) {
        return op->emitOpError()
               << "requires all inputs placed on `devices` attr. The following "
                  "input is placed on device "
               << input_device << " not found in `devices` attr. " << input;
      }
    }
  }

  for (const IfrtArrayType output : outputs) {
    for (const int output_device : output.getDevices()) {
      if (!device_set.count(output_device)) {
        return op->emitOpError()
               << "requires all outputs placed on `devices` attr. The "
                  "following output is placed on device "
               << output_device << " not found in `devices` attr. " << output;
      }
    }
  }

  return mlir::success();
}

struct IoAlias {
  int input_index;
  int output_index;
};

mlir::LogicalResult VerifyPerShardShapesAreEqual(mlir::Operation* op,
                                                 IfrtArrayType t1,
                                                 IfrtArrayType t2) {
  absl::StatusOr<llvm::SmallVector<int64_t>> shard_shape1 =
      t1.getShardingAttr().LocalShapeFromGlobalShape(t1.getShape().getShape());
  if (!shard_shape1.ok()) {
    return op->emitOpError() << "unable to get per-shard shape of array " << t1
                             << ": " << shard_shape1.status().message();
  }
  absl::StatusOr<llvm::SmallVector<int64_t>> shard_shape2 =
      t2.getShardingAttr().LocalShapeFromGlobalShape(t2.getShape().getShape());
  if (!shard_shape2.ok()) {
    return op->emitOpError() << "unable to get per-shard shape of array " << t2
                             << ": " << shard_shape2.status().message();
  }
  if (shard_shape1->size() != shard_shape2->size()) {
    return op->emitOpError()
           << "Arrays have different per-shard shapes: " << t1 << " vs. " << t2;
  }
  for (const auto& [dim1, dim2] : llvm::zip(*shard_shape1, *shard_shape2)) {
    if (dim1 != dim2) {
      return op->emitOpError()
             << "Arrays have different per-shard shapes: " << t1 << " vs. "
             << t2;
    }
  }
  return mlir::success();
}

mlir::LogicalResult VerifyIoAlias(mlir::Operation* op, IoAlias io_alias,
                                  llvm::ArrayRef<IfrtArrayType> inputs,
                                  llvm::ArrayRef<IfrtArrayType> outputs) {
  if (io_alias.input_index < 0 || io_alias.input_index >= inputs.size()) {
    return op->emitOpError()
           << "can't alias input #" << io_alias.input_index << " to output #"
           << io_alias.output_index << " as only having " << inputs.size()
           << " inputs";
  }
  if (io_alias.output_index < 0 || io_alias.output_index >= outputs.size()) {
    return op->emitOpError()
           << "can't alias input #" << io_alias.input_index << " to output #"
           << io_alias.output_index << " as only having " << outputs.size()
           << " outputs";
  }
  if (inputs[io_alias.input_index] != outputs[io_alias.output_index]) {
    // TODO(b/382761415): Relax this aliasing check to allow for different
    // per-shard shapes as long as the byte size is the same. We cannot do this
    // now because we do not have layout information.
    if (inputs[io_alias.input_index].getShape().getElementType() !=
        outputs[io_alias.output_index].getShape().getElementType()) {
      return op->emitOpError()
             << "can't alias input #" << io_alias.input_index << " to output #"
             << io_alias.output_index
             << " with different dtypes: " << inputs[io_alias.input_index]
             << " vs. " << outputs[io_alias.output_index];
    }
    if (mlir::failed(
            VerifyPerShardShapesAreEqual(op, inputs[io_alias.input_index],
                                         outputs[io_alias.output_index]))) {
      return op->emitOpError() << "can't alias input #" << io_alias.input_index
                               << " to output #" << io_alias.output_index;
    }
  }
  return mlir::success();
}

mlir::LogicalResult VerifyIoAliasesAndDonations(
    mlir::Operation* op, mlir::ArrayAttr io_aliases,
    llvm::ArrayRef<int32_t> donated_input_indices,
    llvm::ArrayRef<IfrtArrayType> inputs,
    llvm::ArrayRef<IfrtArrayType> outputs) {
  llvm::SmallSet<int, 4> aliased_or_donated_inputs;
  llvm::SmallSet<int, 4> aliased_outputs;
  for (const int32_t donated_input_index : donated_input_indices) {
    if (donated_input_index < 0 || donated_input_index >= inputs.size()) {
      return op->emitOpError()
             << "can't donate input #" << donated_input_index
             << " as only having " << inputs.size() << " inputs";
    }
    if (!aliased_or_donated_inputs.insert(donated_input_index).second) {
      return op->emitOpError() << "can't donate input #" << donated_input_index
                               << " more than once";
    }
  }
  for (const auto& raw_io_alias :
       io_aliases.getAsRange<mlir::DenseI32ArrayAttr>()) {
    llvm::ArrayRef<int> io_alias_as_array = raw_io_alias.asArrayRef();
    int aliased_input = io_alias_as_array[0];
    int aliased_output = io_alias_as_array[1];
    if (mlir::failed(VerifyIoAlias(op, IoAlias{aliased_input, aliased_output},
                                   inputs, outputs))) {
      return mlir::failure();
    }
    if (!aliased_or_donated_inputs.insert(aliased_input).second) {
      return op->emitOpError() << "can't alias or donate input #"
                               << aliased_input << " more than once";
    }
    if (!aliased_outputs.insert(aliased_output).second) {
      return op->emitOpError()
             << "can't alias output #" << aliased_outputs << " more than once";
    }
  }
  return mlir::success();
}

bool IsAutoLayout(mlir::Type type) {
  auto array = llvm::cast_or_null<IfrtArrayType>(type);
  if (array && array.getLayoutAttr()) {
    return array.getLayoutAttr().str() == "auto";
  }
  return false;
}

bool SameLayout(IfrtArrayType t1, IfrtArrayType t2) {
  xla::LayoutMode layout_mode1 = t1.LayoutMode();
  xla::LayoutMode layout_mode2 = t2.LayoutMode();
  if (layout_mode1.mode != layout_mode2.mode) {
    return false;
  }
  if (layout_mode1.mode == xla::LayoutMode::Mode::kUserSpecified) {
    return layout_mode1.user_layout == layout_mode2.user_layout;
  }
  return true;
}

int GetNumberOfSteps(IfrtIntervalAttr interval) {
  return (interval.getEnd() - interval.getStart() + interval.getStep() - 1) /
         interval.getStep();
}

mlir::LogicalResult CheckIntervalRange(int64_t num_shards,
                                       IfrtIntervalAttr interval,
                                       mlir::Location loc) {
  if (interval.getStart() < 0 || interval.getStart() > num_shards - 1) {
    return mlir::emitError(
        loc, absl::StrCat("start must be in [0, ", num_shards - 1, "], but is ",
                          interval.getStart()));
  }
  if (interval.getStep() <= 0) {
    return mlir::emitError(loc, absl::StrCat("step must be positive, but is ",
                                             interval.getStep()));
  }
  if (interval.getEnd() < 0 ||
      interval.getEnd() > num_shards + interval.getStep() - 1) {
    return mlir::emitError(
        loc, absl::StrCat("end must be in [0, ",
                          num_shards + interval.getStep() - 1, "] if step is ",
                          interval.getStep(), ", but is ", interval.getEnd()));
  }
  return mlir::success();
}

}  // namespace

mlir::LogicalResult ReshardOp::verify() {
  if (getInputs().empty()) {
    return emitOpError() << "requires at least one input array";
  }
  if (getInputs().size() != getOutputs().size()) {
    return emitOpError()
           << "requires the same number of input and output arrays";
  }
  for (const auto [idx, pair] :
       llvm::enumerate(llvm::zip(getInputs(), getOutputs()))) {
    auto input = std::get<0>(pair);
    auto output = std::get<1>(pair);
    if (IsAutoLayout(input.getType()) || IsAutoLayout(output.getType())) {
      return emitOpError()
             << "does not allow input or output arrays with `auto` layout";
    }
    if (mlir::failed(VerifySameGlobalShape(*this, absl::StrCat("input #", idx),
                                           input, absl::StrCat("output #", idx),
                                           output))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult AssembleOp::verify() {
  std::vector<int> input_devices;
  for (const mlir::Value input : getInputs()) {
    const auto array = llvm::cast<IfrtArrayType>(input.getType());
    if (array.getDevices().size() != 1) {
      return emitOpError()
             << "requires every input to be a single device array. Actual: "
             << input.getType();
    }
    if (IsAutoLayout(array)) {
      return emitOpError() << "does not allow input arrays with `auto` layout";
    }
    input_devices.push_back(array.getDevices()[0]);
  }
  const llvm::ArrayRef<int> output_devices = getOutput().getType().getDevices();
  if (!std::equal(input_devices.begin(), input_devices.end(),
                  output_devices.begin())) {
    return emitOpError() << "requires the same input/output device list. Input "
                         << input_devices << " vs Output " << output_devices;
  }
  if (IsAutoLayout(getOutput().getType())) {
    return emitOpError() << "does not allow output arrays with `auto` layout";
  }
  return mlir::success();
}

mlir::LogicalResult DisassembleOp::verify() {
  std::vector<int> output_devices;
  for (const mlir::Value output : getOutputs()) {
    const auto array = llvm::cast<IfrtArrayType>(output.getType());
    if (array.getDevices().size() != 1) {
      return emitOpError()
             << "requires every output to be a single device array. Actual: "
             << output.getType();
    }
    if (IsAutoLayout(array)) {
      return emitOpError() << "does not allow output arrays with `auto` layout";
    }
    output_devices.push_back(array.getDevices()[0]);
  }
  const llvm::ArrayRef<int> input_devices = getInput().getType().getDevices();
  if (!std::equal(input_devices.begin(), input_devices.end(),
                  output_devices.begin())) {
    return emitOpError() << "requires the same input/output device list. Input "
                         << input_devices << " vs Output " << output_devices;
  }
  if (IsAutoLayout(getInput().getType())) {
    return emitOpError() << "does not allow input array with `auto` layout";
  }
  return mlir::success();
}

mlir::LogicalResult CopyArraysOp::verify() {
  int num_in_arrays = getInputs().size();
  int num_out_arrays = getOutputs().size();
  if (num_in_arrays == 0) {
    return emitOpError() << "requires at least one input array";
  }
  if (num_in_arrays != num_out_arrays) {
    return emitOpError()
           << "requires the same number of input and output arrays";
  }
  IfrtArrayType first_input =
      llvm::cast<IfrtArrayType>(getInputs().front().getType());
  auto src_devices = first_input.getDevicesAttr();
  auto src_memory_kind = first_input.MemoryKind();
  IfrtArrayType first_output =
      llvm::cast<IfrtArrayType>(getOutputs().front().getType());
  auto dst_devices = first_output.getDevicesAttr();
  auto dst_memory_kind = first_output.MemoryKind();
  for (const auto [idx, pair] :
       llvm::enumerate(llvm::zip(getInputs(), getOutputs()))) {
    const auto input_array =
        llvm::cast<IfrtArrayType>(std::get<0>(pair).getType());
    if (src_devices != input_array.getDevicesAttr()) {
      return emitOpError() << "requires all input arrays to have the same "
                              "devices, but input #"
                           << idx << " has different devices";
    }
    if (src_memory_kind != input_array.MemoryKind()) {
      return emitOpError() << "requires all input arrays to have the same "
                              "memory kind, but input #"
                           << idx << " has a different memory kind";
    }
    if (IsAutoLayout(input_array)) {
      return emitOpError() << "does not allow input arrays with `auto` layout";
    }
    const auto output_array =
        llvm::cast<IfrtArrayType>(std::get<1>(pair).getType());
    if (dst_devices != output_array.getDevicesAttr()) {
      return emitOpError() << "requires all output arrays to have the same "
                              "devices, but output #"
                           << idx << " has different devices";
    }
    if (dst_memory_kind != output_array.MemoryKind()) {
      return emitOpError() << "requires all output arrays to have the same "
                              "memory kind, but output #"
                           << idx << " has a different memory kind";
    }
    if (IsAutoLayout(output_array)) {
      return emitOpError() << "does not allow output arrays with `auto` layout";
    }
    if (input_array.getShape() != output_array.getShape()) {
      return emitOpError() << "requires input #" << idx << " and output #"
                           << idx << " to have the same shape and dtype";
    }
    // If the sharding is specified, then it should be the same.
    if (!mlir::isa<xla::ifrt::IfrtUnspecifiedShardingAttr>(
            input_array.getShardingAttr()) &&
        !mlir::isa<xla::ifrt::IfrtUnspecifiedShardingAttr>(
            output_array.getShardingAttr()) &&
        input_array.getShardingAttr() != output_array.getShardingAttr()) {
      return emitOpError() << "requires input #" << idx << " and output #"
                           << idx << " to have the same sharding";
    }
  }
  return mlir::success();
}

mlir::LogicalResult RemapArraysOp::verify() {
  int num_in_arrays = getInputs().size();
  int num_out_arrays = getOutputs().size();
  if (num_in_arrays == 0) {
    return emitOpError() << "requires at least one input array";
  }

  std::vector<std::vector<bool>> in_used_shards_list(num_in_arrays);
  for (const auto& [idx, input] : llvm::enumerate(getInputs())) {
    const auto in_array_type = llvm::cast<IfrtArrayType>(input.getType());
    in_used_shards_list[idx].resize(/*count=*/in_array_type.getDevices().size(),
                                    /*value=*/false);
  }
  std::vector<std::vector<bool>> out_mapped_shards_list(num_out_arrays);
  for (const auto& [idx, output] : llvm::enumerate(getOutputs())) {
    const auto out_array_type = llvm::cast<IfrtArrayType>(output.getType());
    out_mapped_shards_list[idx].resize(
        /*count=*/out_array_type.getDevices().size(),
        /*value=*/false);
  }

  auto mappings = getMappings();
  if (mappings.empty()) {
    return emitOpError() << "requires at least one mapping";
  }

  absl::flat_hash_map<int, int> out_array_index_to_in_array_index;
  bool donated = getDonated();
  for (const auto& [idx, array_mapping] : llvm::enumerate(mappings)) {
    const IfrtArrayMappingAttr array_mapping_attr =
        llvm::cast<IfrtArrayMappingAttr>(array_mapping);
    int in_index = array_mapping_attr.getInArrayIndex();
    int out_index = array_mapping_attr.getOutArrayIndex();
    if (in_index < 0 || in_index >= num_in_arrays) {
      return emitOpError() << "mappings in array index " << idx
                           << " must be in [0, " << num_in_arrays
                           << "], but is " << in_index;
    }
    if (out_index < 0 || out_index >= num_out_arrays) {
      return emitOpError() << "mapping out array index " << idx
                           << " must be in [0, " << num_out_arrays
                           << "], but is " << out_index;
    }

    if (!donated) {
      const auto [it, inserted] =
          out_array_index_to_in_array_index.insert({out_index, in_index});
      if (!inserted && it->second != in_index) {
        return emitOpError()
               << "all arguments must be donated because multiple input arrays "
                  "are mapped to output array #"
               << out_index;
      }
    }

    IfrtArrayType in_array_type =
        llvm::cast<IfrtArrayType>(getInputs()[in_index].getType());
    IfrtArrayType out_array_type =
        llvm::cast<IfrtArrayType>(getOutputs()[out_index].getType());
    if (in_array_type.getShape().getElementType() !=
        out_array_type.getShape().getElementType()) {
      return emitOpError() << "requires input array #" << in_index
                           << " and output array #" << out_index
                           << " to have the same dtype: "
                           << in_array_type.getShape().getElementType()
                           << " vs. "
                           << out_array_type.getShape().getElementType();
    }

    if (in_array_type.MemoryKind() != out_array_type.MemoryKind()) {
      return emitOpError() << "requires input array #" << in_index
                           << " and output array #" << out_index
                           << " to have the same memory kind: "
                           << in_array_type.getMemoryKindAttr() << " vs. "
                           << out_array_type.getMemoryKindAttr();
    }

    if (IsAutoLayout(in_array_type) || IsAutoLayout(out_array_type)) {
      return emitOpError()
             << "does not allow input or output arrays with `auto` layout.";
    }

    if (!SameLayout(in_array_type, out_array_type)) {
      return emitOpError() << "requires input array #" << in_index
                           << " and output array #" << out_index
                           << " to have the same layout: "
                           << in_array_type.getLayoutAttr() << " vs. "
                           << out_array_type.getLayoutAttr();
    }

    if (mlir::failed(VerifyPerShardShapesAreEqual(getOperation(), in_array_type,
                                                  out_array_type))) {
      return emitOpError() << "requires input array #" << in_index
                           << " and output array #" << out_index
                           << " to have the same per-shard shape";
    }

    std::vector<bool>& in_used_shards = in_used_shards_list[in_index];
    std::vector<bool>& out_mapped_shards = out_mapped_shards_list[out_index];
    const int64_t in_shards_count = in_used_shards.size();
    const int64_t out_shards_count = out_mapped_shards.size();

    for (const auto& [sidx, mapping] :
         llvm::enumerate(array_mapping_attr.getMappings())) {
      const auto mapping_attr = llvm::cast<IfrtMappingAttr>(mapping);
      IfrtIntervalAttr in_interval = mapping_attr.getFromShards();
      IfrtIntervalAttr out_interval = mapping_attr.getToShards();

      if (mlir::failed(
              CheckIntervalRange(in_shards_count, in_interval, getLoc()))) {
        return emitOpError()
               << "mapping #" << idx << " from #" << sidx << " has invalid "
               << "in_interval: " << mlir::debugString(in_interval);
      }
      if (mlir::failed(
              CheckIntervalRange(out_shards_count, out_interval, getLoc()))) {
        return emitOpError()
               << "mapping #" << idx << " from #" << sidx << " has invalid "
               << "out_interval: " << mlir::debugString(out_interval);
      }

      if (GetNumberOfSteps(in_interval) != GetNumberOfSteps(out_interval)) {
        return emitOpError()
               << "mapping #" << idx << " from #" << sidx << " and mapping #"
               << idx << " to #" << sidx << " must have the same number of "
               << "steps, but were " << GetNumberOfSteps(in_interval) << " and "
               << GetNumberOfSteps(out_interval) << " ("
               << mlir::debugString(in_interval) << " vs. "
               << mlir::debugString(out_interval) << ")";
      }
      int in_shard = in_interval.getStart();
      int out_shard = out_interval.getStart();
      while (in_shard < in_interval.getEnd()) {
        if (in_shard >= in_used_shards.size()) {
          return emitOpError()
                 << "input array #" << in_index << " shard #" << in_shard
                 << " is out of range of input shards.";
        }
        if (out_shard >= out_mapped_shards.size()) {
          return emitOpError()
                 << "output array #" << out_index << " shard #" << out_shard
                 << " is out of range of output shards.";
        }
        if (in_used_shards[in_shard]) {
          return emitOpError() << "input array #" << in_index << " shard #"
                               << in_shard << " is already used.";
        }
        in_used_shards[in_shard] = true;
        if (out_mapped_shards[out_shard]) {
          return emitOpError() << "output array #" << out_index << " shard #"
                               << out_shard << " is already assigned.";
        }
        out_mapped_shards[out_shard] = true;

        in_shard += in_interval.getStep();
        out_shard += out_interval.getStep();
      }
    }
  }

  for (int idx = 0; idx < num_out_arrays; ++idx) {
    for (int out_shard = 0; out_shard < out_mapped_shards_list[idx].size();
         ++out_shard) {
      if (!out_mapped_shards_list[idx][out_shard]) {
        return emitOpError() << "output array #" << idx << " shard #"
                             << out_shard << " is unassigned.";
      }
    }
  }
  return mlir::success();
}

mlir::LogicalResult BitcastArraysOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != outputs.size()) {
    return emitOpError()
           << "requires the same number of input and output arrays. "
           << inputs.size() << " inputs vs. " << outputs.size() << " outputs";
  }
  for (const auto& [idx, input] : llvm::enumerate(inputs)) {
    const auto in_array_type = llvm::cast<IfrtArrayType>(input.getType());
    const auto out_array_type =
        llvm::cast<IfrtArrayType>(outputs[idx].getType());
    // Ideally, the code here would check that the devices are the same.
    // However, since this can be expensive, we only check that the number of
    // devices are the same, and instead rely that an error will be raised
    // at runtime.
    if (in_array_type.getDevices().size() !=
        out_array_type.getDevices().size()) {
      return emitOpError() << "requires input array #" << idx
                           << " and output array #" << idx
                           << " to have the same number of devices: "
                           << in_array_type.getDevices().size() << " vs. "
                           << out_array_type.getDevices().size();
    }
    if (in_array_type.MemoryKind() != out_array_type.MemoryKind()) {
      return emitOpError() << "requires input array #" << idx
                           << " and output array #" << idx
                           << " to have the same memory kind: "
                           << in_array_type.getMemoryKindAttr() << " vs. "
                           << out_array_type.getMemoryKindAttr();
    }
    // TODO(b/382761415): Verify on-device size is the same once we have layout
    // info.
  }
  return mlir::success();
}

mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
}
void CallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
  // Direct call
  if ((*this)->getAttrOfType<mlir::SymbolRefAttr>("callee")) {
    (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
  }
  // Indirect call, callee Value is the first operand.
  return setOperand(0, callee.get<mlir::Value>());
}

mlir::Operation::operand_range CallOp::getArgOperands() { return getInputs(); }
mlir::MutableOperandRange CallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

mlir::LogicalResult CallOp::verifySymbolUses(
    mlir::SymbolTableCollection& symbolTable) {
  mlir::func::FuncOp callee = getCalleeOp(symbolTable);
  mlir::FunctionType callee_type = callee.getFunctionType();
  auto local_view_attr =
      (*this)->getAttrOfType<mlir::UnitAttr>(kIfrtLocalViewAttrName);
  // Verify inputs.
  if (callee_type.getNumInputs() != getInputs().size()) {
    return emitOpError() << "requires the same input size. Input "
                         << getInputs().size() << " vs Callee "
                         << callee_type.getNumInputs();
  }
  for (int i = 0; i < callee_type.getNumInputs(); ++i) {
    if (local_view_attr == nullptr) {
      if (mlir::failed(VerifySameGlobalShape(
              *this, llvm::Twine("Input #").concat(llvm::Twine(i)).str(),
              getInputs()[i], "Callee", callee_type.getInput(i)))) {
        return mlir::failure();
      }
    } else {
      if (mlir::failed(VerifyGlobalLocalShapesEquivalent(
              *this, llvm::Twine("Input #").concat(llvm::Twine(i)).str(),
              getInputs()[i], "Callee", callee_type.getInput(i)))) {
        return mlir::failure();
      }
    }
  }

  // Verify outputs.
  if (callee_type.getNumResults() != getOutputs().size()) {
    return emitOpError() << "requires the same output size. Output "
                         << getOutputs().size() << " vs Callee "
                         << callee_type.getNumResults();
  }
  for (int i = 0; i < callee_type.getNumResults(); ++i) {
    if (local_view_attr == nullptr) {
      if (mlir::failed(VerifySameGlobalShape(
              *this, llvm::Twine("Output #").concat(llvm::Twine(i)).str(),
              getOutputs()[i], "Callee", callee_type.getResult(i)))) {
        return mlir::failure();
      }
    } else {
      if (mlir::failed(VerifyGlobalLocalShapesEquivalent(
              *this, llvm::Twine("Output #").concat(llvm::Twine(i)).str(),
              getOutputs()[i], "Callee", callee_type.getResult(i)))) {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult CallOp::verify() {
  llvm::SmallVector<IfrtArrayType, 4> input_arrays;
  input_arrays.reserve(getInputs().size());
  for (const mlir::Value input : getInputs()) {
    input_arrays.push_back(mlir::cast<IfrtArrayType>(input.getType()));
  }

  llvm::SmallVector<IfrtArrayType, 4> output_arrays;
  output_arrays.reserve(getOutputs().size());
  for (const mlir::Value output : getOutputs()) {
    output_arrays.push_back(mlir::cast<IfrtArrayType>(output.getType()));
  }

  if (mlir::failed(VerifyDevicePlacement(*this, getDevices(), input_arrays,
                                         output_arrays)) ||
      mlir::failed(VerifyIoAliasesAndDonations(*this, getIoAliases(),
                                               getDonatedInputIndices(),
                                               input_arrays, output_arrays))) {
    return mlir::failure();
  }
  return mlir::success();
}

mlir::CallInterfaceCallable CallLoadedExecutableOp::getCallableForCallee() {
  return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
}
void CallLoadedExecutableOp::setCalleeFromCallable(
    mlir::CallInterfaceCallable callee) {
  // Direct call
  if ((*this)->getAttrOfType<mlir::SymbolRefAttr>("callee")) {
    (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
  }
  // Indirect call, callee Value is the first operand.
  return setOperand(0, callee.get<mlir::Value>());
}

mlir::Operation::operand_range CallLoadedExecutableOp::getArgOperands() {
  return getInputs();
}
mlir::MutableOperandRange CallLoadedExecutableOp::getArgOperandsMutable() {
  return getInputsMutable();
}

mlir::LogicalResult CallLoadedExecutableOp::verifySymbolUses(
    mlir::SymbolTableCollection& symbolTable) {
  llvm::SmallVector<mlir::Type, 4> input_types;
  input_types.reserve(getInputs().size());
  for (const mlir::Value input : getInputs()) {
    input_types.push_back(input.getType());
  }
  llvm::SmallVector<mlir::Type, 4> output_types;
  output_types.reserve(getOutputs().size());
  for (const mlir::Value output : getOutputs()) {
    output_types.push_back(output.getType());
  }
  auto func_type =
      mlir::FunctionType::get(getContext(), input_types, output_types);
  LoadedExecutableOp callee = getCalleeOp(symbolTable);
  if (callee.getFunctionType() != func_type) {
    return emitOpError() << "requires callee signature matching " << func_type
                         << ". Actual " << callee.getFunctionType();
  }
  return mlir::success();
}

mlir::LogicalResult CallLoadedExecutableOp::verify() {
  llvm::SmallVector<IfrtArrayType, 4> input_arrays;
  input_arrays.reserve(getInputs().size());
  for (const mlir::Value input : getInputs()) {
    input_arrays.push_back(mlir::cast<IfrtArrayType>(input.getType()));
  }

  llvm::SmallVector<IfrtArrayType, 4> output_arrays;
  output_arrays.reserve(getOutputs().size());
  for (const mlir::Value output : getOutputs()) {
    output_arrays.push_back(mlir::cast<IfrtArrayType>(output.getType()));
  }

  return VerifyIoAliasesAndDonations(*this, getIoAliases(),
                                     getDonatedInputIndices(), input_arrays,
                                     output_arrays);
}

mlir::LogicalResult LoadedExecutableOp::verify() {
  mlir::FunctionType func_type = getFunctionType();

  llvm::SmallVector<IfrtArrayType, 4> input_arrays;
  input_arrays.reserve(func_type.getInputs().size());
  for (const mlir::Type input : func_type.getInputs()) {
    if (auto input_array = llvm::dyn_cast<IfrtArrayType>(input)) {
      input_arrays.push_back(input_array);
    } else {
      return emitOpError() << "requires all inputs to be IfrtArrayType. Found "
                           << input;
    }
  }

  llvm::SmallVector<IfrtArrayType, 4> output_arrays;
  output_arrays.reserve(func_type.getResults().size());
  for (const mlir::Type output : func_type.getResults()) {
    if (auto output_array = llvm::dyn_cast<IfrtArrayType>(output)) {
      output_arrays.push_back(output_array);
    } else {
      return emitOpError() << "requires all outputs to be IfrtArrayType. Found "
                           << output;
    }
  }

  return VerifyDevicePlacement(*this, getDevices(), input_arrays,
                               output_arrays);
}

}  // namespace ifrt
}  // namespace xla
