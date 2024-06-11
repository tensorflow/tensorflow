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
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
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
  } else if (auto array = mlir::dyn_cast<IfrtArrayType>(type)) {
    return array.getShape();
  } else {
    return mlir::failure();
  }
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
    } else {
      return mlir::failure();
    }
  } else {
    // IFRT arrays cannot be in the local view.
    return mlir::failure();
  }
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
    return op->emitOpError()
           << "can't alias input #" << io_alias.input_index << " to output #"
           << io_alias.output_index
           << " with different types: " << inputs[io_alias.input_index]
           << " vs " << outputs[io_alias.output_index];
  }
  return mlir::success();
}

mlir::LogicalResult VerifyIoAliases(mlir::Operation* op,
                                    mlir::ArrayAttr io_aliases,
                                    llvm::ArrayRef<IfrtArrayType> inputs,
                                    llvm::ArrayRef<IfrtArrayType> outputs) {
  llvm::SmallSet<int, 4> aliased_inputs;
  llvm::SmallSet<int, 4> aliased_outputs;
  for (const auto& raw_io_alias :
       io_aliases.getAsRange<mlir::DenseI32ArrayAttr>()) {
    llvm::ArrayRef<int> io_alias_as_array = raw_io_alias.asArrayRef();
    int aliased_input = io_alias_as_array[0];
    int aliased_output = io_alias_as_array[1];
    if (mlir::failed(VerifyIoAlias(op, IoAlias{aliased_input, aliased_output},
                                   inputs, outputs))) {
      return mlir::failure();
    }
    if (!aliased_inputs.insert(aliased_input).second) {
      return op->emitOpError()
             << "can't alias input #" << aliased_input << " more than once";
    }
    if (!aliased_outputs.insert(aliased_output).second) {
      return op->emitOpError()
             << "can't alias output #" << aliased_outputs << " more than once";
    }
  }
  return mlir::success();
}

}  // namespace

mlir::LogicalResult ReshardOp::verify() {
  return VerifySameGlobalShape(*this, "Input", getInput(), "Output",
                               getOutput());
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
    input_devices.push_back(array.getDevices()[0]);
  }
  const llvm::ArrayRef<int> output_devices = getOutput().getType().getDevices();
  if (!std::equal(input_devices.begin(), input_devices.end(),
                  output_devices.begin())) {
    return emitOpError() << "requires the same input/output device list. Input "
                         << input_devices << " vs Output " << output_devices;
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
    output_devices.push_back(array.getDevices()[0]);
  }
  const llvm::ArrayRef<int> input_devices = getInput().getType().getDevices();
  if (!std::equal(input_devices.begin(), input_devices.end(),
                  output_devices.begin())) {
    return emitOpError() << "requires the same input/output device list. Input "
                         << input_devices << " vs Output " << output_devices;
  }
  return mlir::success();
}

mlir::LogicalResult RemapArraysOp::verify() {
  int num_in_arrays = getInputs().size();
  int num_out_arrays = getOutputs().size();
  if (num_in_arrays == 0) {
    return emitOpError() << "requires at least one input array";
  }
  IfrtArrayType first_array =
      llvm::cast<IfrtArrayType>(getInputs()[0].getType());
  mlir::Type dtype = first_array.getShape().getElementType();
  absl::StatusOr<llvm::SmallVector<int64_t>> in_per_shard_shape =
      first_array.getShardingAttr().LocalShapeFromGlobalShape(
          first_array.getShape().getShape());
  if (!in_per_shard_shape.ok()) {
    return emitOpError() << "unable to get per-shard shape of input #0. "
                         << in_per_shard_shape.status().message();
  }
  std::vector<std::vector<bool>> in_used_shards_list(num_in_arrays);
  // Verify that all input/output arrays have the same DType and per-shard
  // shape.
  for (const auto [idx, input] : llvm::enumerate(getInputs())) {
    const auto array = llvm::cast<IfrtArrayType>(input.getType());
    if (array.getShape().getElementType() != dtype) {
      return emitOpError()
             << "requires every input and output array to have the same dtype.";
    }
    auto input_per_shard_shape =
        array.getShardingAttr().LocalShapeFromGlobalShape(
            array.getShape().getShape());
    if (!input_per_shard_shape.ok()) {
      return emitOpError() << "unable to get per-shard shape of input #" << idx
                           << ". " << input_per_shard_shape.status().message();
    }
    if (*input_per_shard_shape != *in_per_shard_shape) {
      return emitOpError() << "requires every input array to have the same "
                              "per-shard shape, but input #"
                           << idx << " has a different shape.";
    }
    in_used_shards_list[idx].resize(/*count=*/array.getDevices().size(),
                                    /*value=*/false);
  }
  std::vector<std::vector<bool>> out_mapped_shards_list(num_out_arrays);
  if (num_out_arrays > 0) {
    IfrtArrayType first_out_array =
        llvm::cast<IfrtArrayType>(getOutputs()[0].getType());
    absl::StatusOr<llvm::SmallVector<int64_t>> out_per_shard_shape =
        first_out_array.getShardingAttr().LocalShapeFromGlobalShape(
            first_out_array.getShape().getShape());
    if (!out_per_shard_shape.ok()) {
      return emitOpError() << "unable to get per-shard shape of output #0. "
                           << out_per_shard_shape.status().message();
    }
    for (const auto [idx, output] : llvm::enumerate(getOutputs())) {
      const auto array = llvm::cast<IfrtArrayType>(output.getType());
      if (array.getShape().getElementType() != dtype) {
        return emitOpError() << "requires every input and output array to have "
                                "the same dtype.";
      }
      auto output_per_shard_shape =
          array.getShardingAttr().LocalShapeFromGlobalShape(
              array.getShape().getShape());
      if (!output_per_shard_shape.ok()) {
        return emitOpError()
               << "unable to get per-shard shape of output #" << idx << ". "
               << output_per_shard_shape.status().message();
      }
      if (*output_per_shard_shape != *out_per_shard_shape) {
        return emitOpError() << "requires every output array to have the same "
                                "per-shard shape, but output #"
                             << idx << " has a different shape.";
      }
      out_mapped_shards_list[idx].resize(/*count=*/array.getDevices().size(),
                                         /*value=*/false);
    }
  }

  // Verify that an input shard is used at most once, and that every output
  // shard has exactly one input shard mapped.
  for (const auto& array_mapping : getMappings()) {
    const auto array_mapping_attr =
        llvm::cast<IfrtArrayMappingAttr>(array_mapping);
    int in_index = array_mapping_attr.getInArrayIndex();
    int out_index = array_mapping_attr.getOutArrayIndex();
    if (in_index < 0 || in_index >= in_used_shards_list.size()) {
      return emitOpError() << "mapping array index " << in_index
                           << " is out of range of input arrays.";
    }
    if (out_index < 0 || out_index >= out_mapped_shards_list.size()) {
      return emitOpError() << "mapping array index " << out_index
                           << " is out of range of output arrays.";
    }
    std::vector<bool>& in_used_shards = in_used_shards_list[in_index];
    std::vector<bool>& out_mapped_shards = out_mapped_shards_list[out_index];
    for (const auto& mapping : array_mapping_attr.getMappings()) {
      const auto mapping_attr = llvm::cast<IfrtMappingAttr>(mapping);
      auto from_shards = mapping_attr.getFromShards();
      auto to_shards = mapping_attr.getToShards();
      int in_shard = from_shards.getStart();
      int out_shard = to_shards.getStart();
      while (in_shard < from_shards.getEnd()) {
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
        in_shard += from_shards.getStep();
        out_shard += to_shards.getStep();
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
      mlir::failed(VerifyIoAliases(*this, getIoAliases(), input_arrays,
                                   output_arrays))) {
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

  return VerifyIoAliases(*this, getIoAliases(), input_arrays, output_arrays);
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
