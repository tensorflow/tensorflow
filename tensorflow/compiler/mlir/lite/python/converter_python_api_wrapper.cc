/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "mlir-c/Bindings/Python/Interop.h"  // from @llvm-project
#include "mlir-c/IR.h"                       // from @llvm-project
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/lib/Bindings/Python/Globals.h"
#include "mlir/lib/Bindings/Python/IRModule.h"
#include "mlir/lib/Bindings/Python/NanobindUtils.h"
#include "mlir/lib/Bindings/Python/Pass.h"
#include "nanobind/nanobind.h"  // from @nanobind
#include "tensorflow/compiler/mlir/lite/python/converter_python_api.h"
#include "tensorflow/compiler/mlir/lite/python/model_utils_core.h"

namespace nb = nanobind;

namespace {

inline nb::object PyoOrThrow(PyObject* ptr) {
  if (PyErr_Occurred() || ptr == nullptr) {
    throw nb::python_error();
  }
  return nb::steal<nb::object>(ptr);
}

}  // namespace

NB_MODULE(_pywrap_converter_api, m) {
  m.def(
      "Convert",
      [](nb::object model_flags_proto_txt_raw,
         nb::object converter_flags_proto_txt_raw,
         nb::object input_contents_txt_raw, bool extended_return,
         nb::object debug_info_txt_raw) {
        return PyoOrThrow(tflite::Convert(
            model_flags_proto_txt_raw.ptr(),
            converter_flags_proto_txt_raw.ptr(), input_contents_txt_raw.ptr(),
            extended_return, debug_info_txt_raw.ptr()));
      },
      nb::arg("model_flags_proto_txt_raw"),
      nb::arg("converter_flags_proto_txt_raw"),
      nb::arg("input_contents_txt_raw") = nb::none(),
      nb::arg("extended_return") = false,
      nb::arg("debug_info_txt_raw") = nb::none(),
      R"pbdoc(
      Convert a model represented in `input_contents`. `model_flags_proto`
      describes model parameters. `flags_proto` describes conversion
      parameters (see relevant .protos for more information). Returns a string
      representing the contents of the converted model. When extended_return
      flag is set to true returns a dictionary that contains string representation
      of the converted model and some statistics like arithmetic ops count.
      `debug_info_str` contains the `GraphDebugInfo` proto.
    )pbdoc");
  m.def(
      "ExperimentalMlirQuantizeModel",
      [](nb::object input_contents_txt_raw, bool disable_per_channel,
         bool fully_quantize, int inference_type, int input_data_type,
         int output_data_type, bool enable_numeric_verify,
         bool enable_whole_model_verify, nb::object op_blocklist,
         nb::object node_blocklist, bool enable_variable_quantization,
         bool disable_per_channel_for_dense_layers,
         nb::object debug_options_proto_txt_raw) {
        return PyoOrThrow(tflite::MlirQuantizeModel(
            input_contents_txt_raw.ptr(), disable_per_channel, fully_quantize,
            inference_type, input_data_type, output_data_type,
            enable_numeric_verify, enable_whole_model_verify,
            op_blocklist.ptr(), node_blocklist.ptr(),
            enable_variable_quantization, disable_per_channel_for_dense_layers,
            debug_options_proto_txt_raw.ptr()));
      },
      nb::arg("input_contents_txt_raw"), nb::arg("disable_per_channel") = false,
      nb::arg("fully_quantize") = true, nb::arg("inference_type") = 9,
      nb::arg("input_data_type") = 0, nb::arg("output_data_type") = 0,
      nb::arg("enable_numeric_verify") = false,
      nb::arg("enable_whole_model_verify") = false,
      nb::arg("op_blocklist") = nb::none(),
      nb::arg("node_blocklist") = nb::none(),
      nb::arg("enable_variable_quantization") = false,
      nb::arg("disable_per_channel_for_dense_layers") = false,
      nb::arg("debug_options_proto_txt_raw") = nullptr,
      R"pbdoc(
      Returns a quantized model.
    )pbdoc");
  m.def(
      "ExperimentalMlirSparsifyModel",
      [](nb::object input_contents_txt_raw) {
        return PyoOrThrow(
            tflite::MlirSparsifyModel(input_contents_txt_raw.ptr()));
      },
      nb::arg("input_contents_txt_raw"),
      R"pbdoc(
      Returns a sparsified model.
    )pbdoc");
  m.def(
      "RegisterCustomOpdefs",
      [](nb::object custom_opdefs_txt_raw) {
        return PyoOrThrow(
            tflite::RegisterCustomOpdefs(custom_opdefs_txt_raw.ptr()));
      },
      nb::arg("custom_opdefs_txt_raw"),
      R"pbdoc(
      Registers the given custom opdefs to the TensorFlow global op registry.
    )pbdoc");
  m.def(
      "RetrieveCollectedErrors",
      []() {
        std::vector<std::string> collected_errors =
            tflite::RetrieveCollectedErrors();
        nb::list serialized_message_list;
        int i = 0;
        for (const auto& error_data : collected_errors) {
          serialized_message_list.append(
              nb::bytes(error_data.data(), error_data.size()));
        }
        return serialized_message_list;
      },
      R"pbdoc(
      Returns and clears the list of collected errors in ErrorCollector.
    )pbdoc");
  m.def(
      "FlatBufferToMlir",
      [](const std::string& model, bool input_is_filepath) {
        return tflite::FlatBufferFileToMlir(model, input_is_filepath);
      },
      R"pbdoc(
      Returns MLIR dump of the given TFLite model.
    )pbdoc");

  m.def("MU_RegisterMlirPasses",
        []() { tflite::model_utils::RegisterMlirPasses(); });
  m.def("MU_RegisterDialects", [](MlirContext context) {
    tflite::model_utils::RegisterDialects(context);
  });
  m.def("MU_CreateIRContext", []() {
    return nb::steal<nb::object>(
        mlirPythonContextToCapsule(tflite::model_utils::CreateIRContext()));
  });
  m.def("MU_FlatBufferToMlir", [](nb::bytes buffer, MlirContext context) {
    return tflite::model_utils::FlatBufferToMlir(
        absl::string_view(static_cast<const char*>(buffer.data()),
                          buffer.size()),
        context);
  });
  m.def("MU_MlirToFlatbuffer", [](MlirOperation op) {
    std::string buffer = tflite::model_utils::MlirToFlatbuffer(op);
    return nb::bytes(buffer.data(), buffer.size());
  });
  m.def("MU_GetOperationAttributeNames", [](MlirOperation op) {
    return tflite::model_utils::GetOperationAttributeNames(op);
  });
  m.def("MU_MlirOpVerify",
        [](MlirOperation op) { return tflite::model_utils::MlirOpVerify(op); });

  auto _mlir = m.def_submodule("_mlir", "MLIR Bindings");
  _mlir.doc() = "MLIR Python Native Extension";

  using namespace mlir::python;
  nb::class_<mlir::python::PyGlobals>(_mlir, "_Globals")
      .def_prop_rw("dialect_search_modules",
                   &PyGlobals::getDialectSearchPrefixes,
                   &PyGlobals::setDialectSearchPrefixes)
      .def("append_dialect_search_prefix", &PyGlobals::addDialectSearchPrefix,
           nb::arg("module_name"))
      .def(
          "_check_dialect_module_loaded",
          [](mlir::python::PyGlobals& self,
             const std::string& dialectNamespace) {
            return self.loadDialectModule(dialectNamespace);
          },
          nb::arg("dialect_namespace"))
      .def("_register_dialect_impl", &PyGlobals::registerDialectImpl,
           nb::arg("dialect_namespace"), nb::arg("dialect_class"),
           "Testing hook for directly registering a dialect")
      .def("_register_operation_impl", &PyGlobals::registerOperationImpl,
           nb::arg("operation_name"), nb::arg("operation_class"), nb::kw_only(),
           nb::arg("replace") = false,
           "Testing hook for directly registering an operation");

  // Aside from making the globals accessible to python, having python manage
  // it is necessary to make sure it is destroyed (and releases its python
  // resources) properly.
  _mlir.attr("globals") =
      nb::cast(new mlir::python::PyGlobals, nb::rv_policy::take_ownership);

  // Registration decorators.
  _mlir.def(
      "register_dialect",
      [](nb::type_object pyClass) {
        std::string dialectNamespace =
            nanobind::cast<std::string>(pyClass.attr("DIALECT_NAMESPACE"));
        mlir::python::PyGlobals::get().registerDialectImpl(dialectNamespace,
                                                           pyClass);
        return pyClass;
      },
      nb::arg("dialect_class"),
      "Class decorator for registering a custom Dialect wrapper");
  _mlir.def(
      "register_operation",
      [](const nb::type_object& dialectClass, bool replace) -> nb::object {
        return nb::cpp_function(
            [dialectClass,
             replace](nb::type_object opClass) -> nb::type_object {
              std::string operationName =
                  nanobind::cast<std::string>(opClass.attr("OPERATION_NAME"));
              mlir::python::PyGlobals::get().registerOperationImpl(
                  operationName, opClass, replace);
              // Dict-stuff the new opClass by name onto the dialect class.
              nb::object opClassName = opClass.attr("__name__");
              dialectClass.attr(opClassName) = opClass;
              return opClass;
            });
      },
      nb::arg("dialect_class"), nb::kw_only(), nb::arg("replace") = false,
      "Produce a class decorator for registering an Operation class as part of "
      "a dialect");
  _mlir.def(
      MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> nb::object {
        return nb::cpp_function([mlirTypeID, replace](
                                    nb::callable typeCaster) -> nb::object {
          PyGlobals::get().registerTypeCaster(mlirTypeID, typeCaster, replace);
          return typeCaster;
        });
      },
      nb::arg("typeid"), nb::kw_only(), nb::arg("replace") = false,
      "Register a type caster for casting MLIR types to custom user types.");
  _mlir.def(
      MLIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> nb::object {
        return nb::cpp_function(
            [mlirTypeID, replace](nb::callable valueCaster) -> nb::object {
              PyGlobals::get().registerValueCaster(mlirTypeID, valueCaster,
                                                   replace);
              return valueCaster;
            });
      },
      nb::arg("typeid"), nb::kw_only(), nb::arg("replace") = false,
      "Register a value caster for casting MLIR values to custom user values.");

  // Define and populate IR submodule.
  auto irModule = _mlir.def_submodule("ir", "MLIR IR Bindings");
  populateIRCore(irModule);
  populateIRAffine(irModule);
  populateIRAttributes(irModule);
  populateIRInterfaces(irModule);
  populateIRTypes(irModule);

  // Define and populate PassManager submodule.
  auto passModule =
      _mlir.def_submodule("passmanager", "MLIR Pass Management Bindings");
  populatePassManagerSubmodule(passModule);
}
