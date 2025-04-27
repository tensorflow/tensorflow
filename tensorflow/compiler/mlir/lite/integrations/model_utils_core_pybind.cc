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
#include <Python.h>

#include <cstddef>
#include <memory>
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
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

namespace {

class MlirPythonPass
    : public mlir::PassWrapper<MlirPythonPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit MlirPythonPass(std::string name, std::string description,
                          py::object pyfunc)
      : name_(name), description_(description), pyfunc_(pyfunc) {
    pyfunc.inc_ref();
  }

  ~MlirPythonPass() override = default;

  mlir::StringRef getName() const override { return name_; }
  mlir::StringRef getArgument() const override { return name_; }
  mlir::StringRef getDescription() const override { return description_; }

  void runOnOperation() override {
    auto module_clone = getOperation().clone();
    MlirModule c_module = wrap(module_clone);

    auto py_module = py::cast(c_module);
    auto py_args = py::make_tuple(py_module);
    PyObject* py_pass_ret = PyObject_CallObject(pyfunc_.ptr(), py_args.ptr());

    if (py_pass_ret == nullptr || PyErr_Occurred()) {
      PyErr_PrintEx(0);
      PyErr_Clear();
      signalPassFailure();
      return;
    }
    auto py_new_module_op = py::cast<py::object>(py_pass_ret);
    auto c_new_module_op = py::cast<MlirOperation>(py_new_module_op);
    mlir::Operation* new_module_op = unwrap(c_new_module_op);

    // TODO: Copy attributes from new_module
    getOperation().getBodyRegion().takeBody(new_module_op->getRegion(0));

    module_clone.erase();
  }

 private:
  std::string name_;
  std::string description_;
  py::object pyfunc_;
};

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

PYBIND11_MODULE(model_utils_core_pybind, m) {
  Py_Initialize();

  m.doc() = "LiteRT ModelUtils Core Pybinds";
  // Register passes on load.
  mlir::registerTransformsPasses();
  mlir::func::registerFuncPasses();
  mlir::odml::registerLegalizeStablehloToVhloPass();

  m.def("mlir_opt_main", [](std::vector<std::string> argv,
                            std::vector<std::string> pass_names,
                            std::vector<std::string> pass_descriptions,
                            std::vector<py::object> pass_fns) {
    std::vector<char*> c_argv_vec;
    c_argv_vec.reserve(argv.size());
    for (size_t i = 0; i < argv.size(); ++i)
      c_argv_vec.push_back(const_cast<char*>(argv[i].c_str()));

    int argc = argv.size();
    char** c_argv = c_argv_vec.data();

    tensorflow::InitMlir y(&argc, &c_argv);

    mlir::DialectRegistry registry;
    RegisterDialects(registry);

    int num_passes = pass_names.size();
    for (int i = 0; i < num_passes; ++i) {
      mlir::PassRegistration<MlirPythonPass>(
          [&, i = i]() -> std::unique_ptr<mlir::Pass> {
            std::unique_ptr<mlir::Pass> p = std::make_unique<MlirPythonPass>(
                pass_names[i], pass_descriptions[i], pass_fns[i]);
            return p;
          });
    }

    (void)mlir::MlirOptMain(argc, c_argv, "ModelUtils python passes driver\n",
                            registry);
  });

  m.def("register_dialects", [](MlirContext context) {
    mlir::DialectRegistry registry;
    RegisterDialects(registry);
    unwrap(context)->appendDialectRegistry(registry);
    unwrap(context)->loadAllAvailableDialects();
  });

  m.def("flatbuffer_to_mlir",
        [](py::bytes buffer, MlirContext context) -> MlirModule {
          mlir::DialectRegistry registry;
          RegisterDialects(registry);
          unwrap(context)->appendDialectRegistry(registry);
          unwrap(context)->loadAllAvailableDialects();

          auto module_op = tflite::FlatBufferToMlir(
              buffer, unwrap(context), mlir::UnknownLoc::get(unwrap(context)));
          return wrap(module_op.release());
        });

  m.def("mlir_to_flatbuffer", [](MlirOperation c_op) {
    auto op = unwrap(c_op);
    auto module_op = llvm::dyn_cast<mlir::ModuleOp>(op);

    tflite::FlatbufferExportOptions options;
    std::string result;
    tflite::MlirToFlatBufferTranslateFunction(module_op, options, &result,
                                              true);
    return py::bytes(result);
  });

  m.def("get_operation_attribute_names", [](MlirOperation c_op) {
    mlir::Operation* op = unwrap(c_op);

    std::vector<std::string> attr_names;
    for (auto attr : op->getAttrDictionary()) {
      attr_names.push_back(attr.getName().str());
    }
    return attr_names;
  });

  m.def("get_dictionary_attr_names", [](MlirAttribute c_attr) {
    auto attr = mlir::cast<mlir::DictionaryAttr>(unwrap(c_attr));
    std::vector<std::string> attr_names;
    for (auto attr : attr) {
      attr_names.push_back(attr.getName().str());
    }
    return attr_names;
  });

  m.def("get_elements_attr_buffer", [](MlirAttribute c_attr) {
    auto attr = mlir::cast<mlir::ElementsAttr>(unwrap(c_attr));

    tensorflow::Tensor tensor;
    auto status = tensorflow::ConvertToTensor(attr, &tensor);
    PyObject* np_array = Py_None;
    status = tensorflow::TensorToNdarray(tensor, &np_array);

    return py::reinterpret_steal<py::object>(np_array);
  });
}

}  // namespace
