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

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
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
#include "stablehlo/dialect/StablehloOps.h"
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
  if (llvm::isa<mlir::stablehlo::TokenType>(type)) {
    return mlir::RankedTensorType::get({},
                                       IfrtTokenType::get(type.getContext()));
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
  if (llvm::isa<mlir::stablehlo::TokenType>(type)) {
    return mlir::RankedTensorType::get({},
                                       IfrtTokenType::get(type.getContext()));
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

// Collects the `IfrtArrayType` of each value in `values`.
llvm::SmallVector<IfrtArrayType, 4> CollectArrayTypes(mlir::ValueRange values) {
  llvm::SmallVector<IfrtArrayType, 4> arrays;
  arrays.reserve(values.size());
  for (const mlir::Value value : values) {
    arrays.push_back(GetArrayType(value));
  }
  return arrays;
}

bool IsAutoLayout(mlir::Type type) {
  auto array = llvm::cast_or_null<IfrtArrayType>(type);
  return array != nullptr && array.getLayoutAttr().getValue() == "auto";
}

// Verifies that `in_array` (described by `in_desc`) and `out_array` (described
// by `out_desc`) have identical element types, memory kinds, concrete layouts,
// and per-shard shapes.
mlir::LogicalResult VerifySameShardArrayMetadata(mlir::Operation* op,
                                                 llvm::StringRef in_desc,
                                                 IfrtArrayType in_array,
                                                 llvm::StringRef out_desc,
                                                 IfrtArrayType out_array) {
  if (IsAutoLayout(in_array)) {
    return op->emitOpError()
           << "does not allow " << in_desc << " with `auto` layout";
  }
  if (IsAutoLayout(out_array)) {
    return op->emitOpError()
           << "does not allow " << out_desc << " with `auto` layout";
  }
  if (in_array.getShape().getElementType() !=
      out_array.getShape().getElementType()) {
    return op->emitOpError() << "requires " << in_desc << " and " << out_desc
                             << " to have the same dtype: "
                             << in_array.getShape().getElementType() << " vs. "
                             << out_array.getShape().getElementType();
  }
  if (!in_array.HasSameMemoryKind(out_array)) {
    return op->emitOpError()
           << "requires " << in_desc << " and " << out_desc
           << " to have the same memory kind: " << in_array.getMemoryKindAttr()
           << " vs. " << out_array.getMemoryKindAttr();
  }
  if (!in_array.HasSameLayout(out_array)) {
    return op->emitOpError()
           << "requires " << in_desc << " and " << out_desc
           << " to have the same layout: " << in_array.getLayoutAttr()
           << " vs. " << out_array.getLayoutAttr();
  }
  if (mlir::failed(VerifyPerShardShapesAreEqual(op, in_array, out_array))) {
    return op->emitOpError() << "requires " << in_desc << " and " << out_desc
                             << " to have the same per-shard shape";
  }
  return mlir::success();
}

mlir::LogicalResult CheckIntervalRange(int64_t num_shards,
                                       IfrtIntervalAttr interval,
                                       mlir::Location loc) {
  if (interval.getStart() >= num_shards) {
    return mlir::emitError(
        loc, absl::StrCat("start must be in [0, ", num_shards - 1, "], but is ",
                          interval.getStart()));
  }
  const int64_t last_index =
      interval.getEnd() - 1 -
      (interval.getEnd() - 1 - interval.getStart()) % interval.getStep();
  if (last_index >= num_shards) {
    return mlir::emitError(
        loc, absl::StrCat("last index in interval must be in [0, ",
                          num_shards - 1, "], but is ", last_index));
  }
  return mlir::success();
}

}  // namespace

mlir::LogicalResult ReshardOp::verify() {
  for (const auto [idx, pair] :
       llvm::enumerate(llvm::zip(getInputs(), getOutputs()))) {
    mlir::Value input = std::get<0>(pair);
    mlir::Value output = std::get<1>(pair);
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
  input_devices.reserve(getInputs().size());
  for (const mlir::Value input : getInputs()) {
    const IfrtArrayType array = GetArrayType(input);
    if (array.getDevices().size() != 1) {
      return emitOpError()
             << "requires every input to be a single device array. Actual: "
             << input.getType();
    }
    input_devices.push_back(array.getDevices().front());
  }
  const IfrtArrayType output_array = getOutput().getType();
  const llvm::ArrayRef<int> output_devices = output_array.getDevices();
  if (!llvm::equal(input_devices, output_devices)) {
    return emitOpError() << "requires the same input/output device list. Input "
                         << input_devices << " vs Output " << output_devices;
  }
  for (const auto& [idx, input] : llvm::enumerate(getInputs())) {
    if (mlir::failed(VerifySameShardArrayMetadata(
            getOperation(), absl::StrCat("input array #", idx),
            GetArrayType(input), "output array", output_array))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult DisassembleOp::verify() {
  std::vector<int> output_devices;
  output_devices.reserve(getOutputs().size());
  for (const mlir::Value output : getOutputs()) {
    const IfrtArrayType array = GetArrayType(output);
    if (array.getDevices().size() != 1) {
      return emitOpError()
             << "requires every output to be a single device array. Actual: "
             << output.getType();
    }
    output_devices.push_back(array.getDevices().front());
  }
  const IfrtArrayType input_array = getInput().getType();
  const llvm::ArrayRef<int> input_devices = input_array.getDevices();
  if (!llvm::equal(input_devices, output_devices)) {
    return emitOpError() << "requires the same input/output device list. Input "
                         << input_devices << " vs Output " << output_devices;
  }
  for (const auto& [idx, output] : llvm::enumerate(getOutputs())) {
    if (mlir::failed(VerifySameShardArrayMetadata(
            getOperation(), "input array", input_array,
            absl::StrCat("output array #", idx), GetArrayType(output)))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult CopyArraysOp::verify() {
  if (getDonated() && getReuse()) {
    return emitOpError()
           << "requires at most one of `donated` or `reuse` to be "
              "set to true";
  }
  IfrtArrayType first_input = GetArrayType(getInputs().front());
  IfrtDevicesAttr src_devices = first_input.getDevicesAttr();
  mlir::StringAttr src_memory_kind = first_input.getMemoryKindAttr();
  IfrtArrayType first_output = GetArrayType(getOutputs().front());
  IfrtDevicesAttr dst_devices = first_output.getDevicesAttr();
  mlir::StringAttr dst_memory_kind = first_output.getMemoryKindAttr();
  for (const auto [idx, pair] :
       llvm::enumerate(llvm::zip(getInputs(), getOutputs()))) {
    const IfrtArrayType input_array = GetArrayType(std::get<0>(pair));
    if (src_devices != input_array.getDevicesAttr()) {
      return emitOpError() << "requires all input arrays to have the same "
                              "devices, but input #"
                           << idx << " has different devices";
    }
    if (src_memory_kind != input_array.getMemoryKindAttr()) {
      return emitOpError() << "requires all input arrays to have the same "
                              "memory kind, but input #"
                           << idx << " has a different memory kind";
    }
    if (IsAutoLayout(input_array)) {
      return emitOpError() << "does not allow input arrays with `auto` layout";
    }
    const IfrtArrayType output_array = GetArrayType(std::get<1>(pair));
    if (dst_devices != output_array.getDevicesAttr()) {
      return emitOpError() << "requires all output arrays to have the same "
                              "devices, but output #"
                           << idx << " has different devices";
    }
    if (dst_memory_kind != output_array.getMemoryKindAttr()) {
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
  if (num_out_arrays == 0) {
    return emitOpError() << "requires at least one output array";
  }

  std::vector<std::vector<bool>> in_used_shards_list(num_in_arrays);
  for (const auto& [idx, input] : llvm::enumerate(getInputs())) {
    const IfrtArrayType in_array_type = GetArrayType(input);
    if (IsAutoLayout(in_array_type)) {
      return emitOpError() << "does not allow input array #" << idx
                           << " with `auto` layout";
    }
    in_used_shards_list[idx].resize(/*count=*/in_array_type.getDevices().size(),
                                    /*value=*/false);
  }
  std::vector<std::vector<bool>> out_mapped_shards_list(num_out_arrays);
  for (const auto& [idx, output] : llvm::enumerate(getOutputs())) {
    const IfrtArrayType out_array_type = GetArrayType(output);
    if (IsAutoLayout(out_array_type)) {
      return emitOpError() << "does not allow output array #" << idx
                           << " with `auto` layout";
    }
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
    if (in_index >= num_in_arrays) {
      return emitOpError() << "mappings in array index " << idx
                           << " must be in [0, " << num_in_arrays
                           << "], but is " << in_index;
    }
    if (out_index >= num_out_arrays) {
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

    IfrtArrayType in_array_type = GetArrayType(getInputs()[in_index]);
    IfrtArrayType out_array_type = GetArrayType(getOutputs()[out_index]);
    if (mlir::failed(VerifySameShardArrayMetadata(
            getOperation(), absl::StrCat("input array #", in_index),
            in_array_type, absl::StrCat("output array #", out_index),
            out_array_type))) {
      return mlir::failure();
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

      int64_t in_shard = in_interval.getStart();
      int64_t out_shard = out_interval.getStart();
      while (in_shard < in_interval.getEnd()) {
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
  if (!getDonated()) {
    return emitOpError() << "requires `donated` to be set to true";
  }
  for (const auto& [idx, pair] :
       llvm::enumerate(llvm::zip(getInputs(), getOutputs()))) {
    const IfrtArrayType in_array_type = GetArrayType(std::get<0>(pair));
    const IfrtArrayType out_array_type = GetArrayType(std::get<1>(pair));
    if (IsAutoLayout(in_array_type) || IsAutoLayout(out_array_type)) {
      return emitOpError()
             << "does not allow input or output arrays with `auto` layout";
    }
    if (in_array_type.getDevicesAttr() != out_array_type.getDevicesAttr()) {
      return emitOpError() << "requires input array #" << idx
                           << " and output array #" << idx
                           << " to have the same devices: "
                           << in_array_type.getDevicesAttr() << " vs. "
                           << out_array_type.getDevicesAttr();
    }
    if (!in_array_type.HasSameMemoryKind(out_array_type)) {
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
  return getCalleeAttr();
}
void CallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
  // Direct call.
  if (getCalleeAttr() != nullptr) {
    setCalleeAttr(mlir::cast<mlir::SymbolRefAttr>(callee));
    return;
  }
  // Indirect call, callee Value is the first operand.
  setOperand(0, mlir::cast<mlir::Value>(callee));
}

mlir::Operation::operand_range CallOp::getArgOperands() { return getInputs(); }
mlir::MutableOperandRange CallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

mlir::LogicalResult CallOp::verifySymbolUses(
    mlir::SymbolTableCollection& symbolTable) {
  mlir::func::FuncOp callee = getCalleeOp(symbolTable);
  mlir::FunctionType callee_type = callee.getFunctionType();
  const bool is_local_view = (*this)->hasAttr(kIfrtLocalViewAttrName);
  // Verify inputs.
  if (callee_type.getNumInputs() != getInputs().size()) {
    return emitOpError() << "requires the same input size. Input "
                         << getInputs().size() << " vs Callee "
                         << callee_type.getNumInputs();
  }
  for (int i = 0; i < callee_type.getNumInputs(); ++i) {
    if (!is_local_view) {
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
    if (!is_local_view) {
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
  llvm::SmallVector<IfrtArrayType, 4> input_arrays =
      CollectArrayTypes(getInputs());
  llvm::SmallVector<IfrtArrayType, 4> output_arrays =
      CollectArrayTypes(getOutputs());

  return VerifyDevicePlacement(*this, getDevices(), input_arrays,
                               output_arrays);
}

mlir::CallInterfaceCallable CallLoadedExecutableOp::getCallableForCallee() {
  return getCalleeAttr();
}
void CallLoadedExecutableOp::setCalleeFromCallable(
    mlir::CallInterfaceCallable callee) {
  // Direct call.
  if (getCalleeAttr() != nullptr) {
    setCalleeAttr(mlir::cast<mlir::SymbolRefAttr>(callee));
    return;
  }
  // Indirect call, callee Value is the first operand.
  setOperand(0, mlir::cast<mlir::Value>(callee));
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

mlir::LogicalResult AfterOp::verify() {
  if (getInputs().empty()) {
    return emitOpError() << "requires at least one input array";
  }
  return mlir::success();
}

}  // namespace ifrt
}  // namespace xla
