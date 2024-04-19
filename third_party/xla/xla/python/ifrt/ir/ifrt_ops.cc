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
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
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
  if (auto ranked_tensor = type.dyn_cast<mlir::RankedTensorType>()) {
    return ranked_tensor;
  } else if (auto array = type.dyn_cast<IfrtArrayType>()) {
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
  if (auto local_ranked_tensor = type.dyn_cast<mlir::RankedTensorType>()) {
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
  auto array = call_value.getType().dyn_cast<IfrtArrayType>();
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
  llvm::SmallSet<int, 4> device_set;
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
  llvm::SmallVector<int, 4> input_devices;
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
  llvm::SmallVector<int, 4> output_devices;
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
    mlir::SymbolTableCollection& symbol_table) {
  mlir::func::FuncOp callee = getCalleeOp(symbol_table);
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
    input_arrays.push_back(input.getType().cast<IfrtArrayType>());
  }

  llvm::SmallVector<IfrtArrayType, 4> output_arrays;
  output_arrays.reserve(getOutputs().size());
  for (const mlir::Value output : getOutputs()) {
    output_arrays.push_back(output.getType().cast<IfrtArrayType>());
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
    mlir::SymbolTableCollection& symbol_table) {
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
  LoadedExecutableOp callee = getCalleeOp(symbol_table);
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
    input_arrays.push_back(input.getType().cast<IfrtArrayType>());
  }

  llvm::SmallVector<IfrtArrayType, 4> output_arrays;
  output_arrays.reserve(getOutputs().size());
  for (const mlir::Value output : getOutputs()) {
    output_arrays.push_back(output.getType().cast<IfrtArrayType>());
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
