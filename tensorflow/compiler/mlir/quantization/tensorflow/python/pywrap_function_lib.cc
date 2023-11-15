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
#include <string>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/type_casters.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace py = ::pybind11;

namespace {

using ::tensorflow::SignatureDef;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PyFunctionLibrary;
using ::tensorflow::quantization::QuantizationOptions;

// A "trampoline" class that redirects virtual function calls to the python
// implementation.
//
// Reference:
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
class PyFunctionLibraryTrampoline : public PyFunctionLibrary {
 public:
  using PyFunctionLibrary::PyFunctionLibrary;

  ExportedModel AssignIdsToCustomAggregatorOps(
      const ExportedModel& exported_model) const override {
    PYBIND11_OVERRIDE_PURE(ExportedModel, PyFunctionLibrary,
                           assign_ids_to_custom_aggregator_ops, exported_model);
  }

  void SaveExportedModel(const absl::string_view dst_saved_model_path,
                         const ExportedModel& exported_model,
                         const absl::string_view src_saved_model_path,
                         const std::unordered_set<std::string>& tags,
                         const absl::flat_hash_map<std::string, SignatureDef>&
                             signature_def_map) const override {
    PYBIND11_OVERRIDE_PURE(void, PyFunctionLibrary, save_exported_model,
                           dst_saved_model_path, exported_model,
                           src_saved_model_path, tags, signature_def_map);
  }

  ExportedModel RunCalibration(
      const absl::string_view saved_model_path,
      const ExportedModel& exported_model,
      const QuantizationOptions& quantization_options,
      const py::object representative_dataset) const override {
    PYBIND11_OVERRIDE_PURE(ExportedModel, PyFunctionLibrary, run_calibration,
                           saved_model_path, exported_model,
                           quantization_options, representative_dataset);
  }
};

}  // namespace

PYBIND11_MODULE(pywrap_function_lib, m) {
  py::class_<PyFunctionLibrary, PyFunctionLibraryTrampoline>(
      m, "PyFunctionLibrary")
      .def(py::init<>())
      .def("assign_ids_to_custom_aggregator_ops",
           &PyFunctionLibrary::AssignIdsToCustomAggregatorOps,
           py::arg("exported_model_serialized"))
      .def("save_exported_model", &PyFunctionLibrary::SaveExportedModel,
           py::arg("dst_saved_model_path"),
           py::arg("exported_model_serialized"),
           py::arg("src_saved_model_path"), py::arg("tags"),
           py::arg("serialized_signature_def_map"))
      .def("run_calibration", &PyFunctionLibrary::RunCalibration,
           py::arg("saved_model_path"), py::arg("exported_model_serialized"),
           py::arg("quantization_options_serialized"),
           py::arg("representative_dataset"));
}
