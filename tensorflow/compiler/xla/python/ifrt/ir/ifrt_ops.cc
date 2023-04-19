/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/ir/ifrt_ops.h"

#include <algorithm>
#include <cstdint>

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
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/python/ifrt/ir/ifrt_dialect.h"

// Generated definitions.
#define GET_OP_CLASSES
#include "tensorflow/compiler/xla/python/ifrt/ir/ifrt_ops.cc.inc"

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

}  // namespace

mlir::LogicalResult ReshardOp::verify() {
  return VerifySameGlobalShape(*this, "Input", getInput(), "Output",
                               getOutput());
}

mlir::LogicalResult AssembleOp::verify() {
  llvm::SmallVector<int64_t, 4> input_devices;
  for (const mlir::Value input : getInputs()) {
    const auto array = llvm::cast<IfrtArrayType>(input.getType());
    if (array.getDevices().size() != 1) {
      return emitOpError()
             << "requires every input to be a single device array. Actual: "
             << input.getType();
    }
    input_devices.push_back(array.getDevices()[0]);
  }
  const llvm::ArrayRef<int64_t> output_devices =
      getOutput().getType().getDevices();
  if (!std::equal(input_devices.begin(), input_devices.end(),
                  output_devices.begin())) {
    return emitOpError() << "requires the same input/output device list. Input "
                         << input_devices << " vs Output " << output_devices;
  }
  return mlir::success();
}

mlir::LogicalResult DisassembleOp::verify() {
  llvm::SmallVector<int64_t, 4> output_devices;
  for (const mlir::Value output : getOutputs()) {
    const auto array = llvm::cast<IfrtArrayType>(output.getType());
    if (array.getDevices().size() != 1) {
      return emitOpError()
             << "requires every output to be a single device array. Actual: "
             << output.getType();
    }
    output_devices.push_back(array.getDevices()[0]);
  }
  const llvm::ArrayRef<int64_t> input_devices =
      getInput().getType().getDevices();
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

mlir::Operation::operand_range CallOp::getArgOperands() { return getInputs(); }

mlir::LogicalResult CallOp::verifySymbolUses(
    mlir::SymbolTableCollection& symbolTable) {
  const auto callee_attr =
      (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
  if (!callee_attr) {
    return emitOpError() << "requires `callee` SymbolRefAttr";
  }
  auto callee = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(
      *this, callee_attr);
  if (!callee) {
    return emitOpError() << "requires '" << callee_attr
                         << "' to reference a valid function";
  }
  mlir::FunctionType callee_type = callee.getFunctionType();

  // Verify inputs.
  if (callee_type.getNumInputs() != getInputs().size()) {
    return emitOpError() << "requires the same input size. Input "
                         << getInputs().size() << " vs Callee "
                         << callee_type.getNumInputs();
  }
  for (int i = 0; i < callee_type.getNumInputs(); ++i) {
    if (mlir::failed(VerifySameGlobalShape(
            *this, llvm::Twine("Input #").concat(llvm::Twine(i)).str(),
            getInputs()[i], "Callee", callee_type.getInput(i)))) {
      return mlir::failure();
    }
  }

  // Verify outputs.
  if (callee_type.getNumResults() != getOutputs().size()) {
    return emitOpError() << "requires the same output size. Output "
                         << getOutputs().size() << " vs Callee "
                         << callee_type.getNumResults();
  }
  for (int i = 0; i < callee_type.getNumResults(); ++i) {
    if (mlir::failed(VerifySameGlobalShape(
            *this, llvm::Twine("Output #").concat(llvm::Twine(i)).str(),
            getOutputs()[i], "Callee", callee_type.getResult(i)))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult CallOp::verify() {
  llvm::SmallSet<int64_t, 4> attr_devices;
  for (const int64_t device : getDevices()) {
    if (!attr_devices.insert(device).second) {
      return emitOpError() << "has duplicated device id " << device
                           << " in `devices` attr";
    }
  }

  for (const mlir::Value input : getInputs()) {
    for (const int64_t input_device :
         input.getType().cast<IfrtArrayType>().getDevices()) {
      if (!attr_devices.count(input_device)) {
        return emitOpError()
               << "requires all inputs placed on `devices` attr. The following "
                  "input is placed on device "
               << input_device << " not found in `devices` attr. "
               << input.getType();
      }
    }
  }

  for (const mlir::Value output : getOutputs()) {
    for (const int64_t output_device :
         output.getType().cast<IfrtArrayType>().getDevices()) {
      if (!attr_devices.count(output_device)) {
        return emitOpError()
               << "requires all outputs placed on `devices` attr. The "
                  "following output is placed on device "
               << output_device << " not found in `devices` attr. "
               << output.getType();
      }
    }
  }

  return mlir::success();
}

}  // namespace ifrt
}  // namespace xla
