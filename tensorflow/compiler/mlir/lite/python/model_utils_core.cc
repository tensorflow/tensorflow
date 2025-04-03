/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/python/model_utils_core.h"

#include <Python.h>

#include <string>
#include <vector>

#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/Casting.h"
#include "mlir-c/IR.h"  // from @llvm-project
#include "mlir/Bindings/Python/PybindAdaptors.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/CAPI/IR.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

namespace tflite {
namespace model_utils {

namespace py = pybind11;

namespace {

inline void RegisterDialects(mlir::DialectRegistry& registry) {
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::quant::QuantDialect,
                  mlir::quantfork::QuantizationForkDialect,
                  mlir::TFL::TensorFlowLiteDialect,
                  mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
}

void RegisterMlirPasses() {
  mlir::registerTransformsPasses();
  mlir::func::registerFuncPasses();
  mlir::registerPass(
      []() { return mlir::odml::createLegalizeStablehloToVhloPass(); });
}

void RegisterDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

MlirContext CreateIRContext() {
  MlirContext ctx = mlirContextCreate();
  RegisterDialects(ctx);
  return ctx;
}

MlirModule FlatBufferToMlir(py::bytes buffer, MlirContext context) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();

  auto module_op = tflite::FlatBufferToMlir(
      buffer, unwrap(context), mlir::UnknownLoc::get(unwrap(context)));
  return wrap(module_op.release());
}

py::bytes MlirToFlatbuffer(MlirOperation c_op) {
  auto op = unwrap(c_op);
  auto module_op = llvm::dyn_cast<mlir::ModuleOp>(op);

  tflite::FlatbufferExportOptions options;
  std::string result;
  tflite::MlirToFlatBufferTranslateFunction(module_op, options, &result, true);
  return py::bytes(result);
}

std::vector<std::string> GetOperationAttributeNames(MlirOperation c_op) {
  mlir::Operation* op = unwrap(c_op);

  std::vector<std::string> attr_names;
  for (auto attr : op->getAttrDictionary()) {
    attr_names.push_back(attr.getName().str());
  }
  return attr_names;
}

// absl::StatusOr<tensorflow::Tensor> GetElementsAttrTensor(MlirAttribute
// c_attr) {
//   auto attr = mlir::cast<mlir::ElementsAttr>(unwrap(c_attr));

//   tensorflow::Tensor tensor;
//   auto status = tensorflow::ConvertToTensor(attr, &tensor);
//   if (!status.ok()) {
//     return status;
//   }
//   return tensor;
// }

}  // namespace
}  // namespace model_utils
}  // namespace tflite
