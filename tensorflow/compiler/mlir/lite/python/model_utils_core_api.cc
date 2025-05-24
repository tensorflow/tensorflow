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
#include "tensorflow/compiler/mlir/lite/python/model_utils_core_api.h"

#include <any>
#include <cstddef>
#include <string>
#include <typeinfo>
#include <utility>

#include "absl/log/log.h"
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
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"

namespace py = pybind11;

namespace {

class MuContainer {
 public:
  template <typename T>
  explicit MuContainer(T&& d) : data_(std::any(std::forward<T>(d))) {}

  template <typename T, typename... Args>
  static MuContainer create(Args&&... args) {
    return MuContainer(std::make_any<T>(std::forward<Args>(args)...));
  }

  template <typename T>
  inline T* get() {
    return std::any_cast<T>(&data_);
  }

 private:
  std::any data_;
};

template <typename T>
class MlirObject : public MuContainer {
 public:
  explicit MlirObject(T&& data) : MuContainer(std::forward<T>(data)) {}

  inline T* get() { return MuContainer::get<T>(); }

  std::string to_string() {
    std::string s;
    llvm::raw_string_ostream ostream(s);
    get()->print(ostream);
    return s;
  }
};

using MuMlirAttribute = MlirObject<mlir::Attribute>;
using MuModuleOp = MlirObject<mlir::ModuleOp>;
// using MuStringAttr = MlirObject<mlir::StringAttr>;

}  // namespace

void PopulateModelUtilsCoreApis(py::module& m_) {
  auto mu = m_.def_submodule("model_utils_core_api");
  py::class_<MuContainer>(mu, "MuContainer");

#define DEFINE_PY_MLIR_OBJECT(name) \
  py::class_<name>(mu, #name).def(  \
      "__str__", [](name& self) { return self.to_string(); });

  DEFINE_PY_MLIR_OBJECT(MuModuleOp);
  DEFINE_PY_MLIR_OBJECT(MuMlirAttribute);

  mu.def("create_ir_context", []() {
    auto context = std::make_unique<mlir::MLIRContext>();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
    mlir::func::registerAllExtensions(registry);
    registry.insert<
        mlir::arith::ArithDialect, mlir::func::FuncDialect,
        mlir::quant::QuantDialect, mlir::quantfork::QuantizationForkDialect,
        mlir::TFL::TensorFlowLiteDialect, mlir::stablehlo::StablehloDialect,
        mlir::vhlo::VhloDialect>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();

    context->disableMultithreading();
    context->allowUnregisteredDialects();

    return MuContainer(context.release());
  });

  mu.def("flatbuffer_to_mlir", [](py::bytes buffer, MuContainer* context_) {
    mlir::MLIRContext* context = *context_->get<mlir::MLIRContext*>();

    mlir::ModuleOp module_op =
        tflite::FlatBufferToMlir(buffer, context,
                                 mlir::UnknownLoc::get(context))
            .release();

    return MuModuleOp(std::move(module_op));
  });

  mu.def("mlir_to_flatbuffer", [](MuModuleOp* module_op_) {
    mlir::ModuleOp module_op = *module_op_->get();

    tflite::FlatbufferExportOptions options;
    std::string result;
    tflite::MlirToFlatBufferTranslateFunction(module_op, options, &result,
                                              true);
    return py::bytes(result);
  });

  mu.def("register_StringAttr", [](py::object cls) {
    py::setattr(cls, "_typeid", py::cpp_function([]() {
                  return typeid(mlir::StringAttr).hash_code();
                }));

    py::setattr(cls, "_to_mlir",
                py::cpp_function([](py::object self, MuContainer* context_) {
                  mlir::MLIRContext* context =
                      *context_->get<mlir::MLIRContext*>();
                  py::str py_str = self.attr("data");
                  mlir::StringAttr attr = mlir::StringAttr::get(
                      context, py_str.cast<std::string>());
                  return MuMlirAttribute(std::move(attr));
                }));

    py::setattr(cls, "_from_mlir",
                py::cpp_function([cls](MuMlirAttribute* attr_) {
                  auto attr = mlir::dyn_cast<mlir::StringAttr>(*attr_->get());
                  return cls(py::str(attr.strref().str()));
                }));
  });
}
