/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <string>

#include "pybind11/functional.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/lite/python/optimize/calibration_wrapper.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace py = pybind11;
using tflite::calibration_wrapper::AddIntermediateTensors;
using tflite::calibration_wrapper::CalibrationWrapper;

PYBIND11_MODULE(_pywrap_tensorflow_lite_calibration_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_tensorflow_lite_calibration_wrapper
    -----
  )pbdoc";
  m.def("AddIntermediateTensors", [](py::handle& data) {
    return tensorflow::PyoOrThrow(AddIntermediateTensors(data.ptr()));
  });
  py::class_<CalibrationWrapper>(m, "CalibrationWrapper")
      .def(py::init([](py::handle& data,
                       const std::vector<std::string>& registerers_by_name,
                       const std::vector<std::function<void(uintptr_t)>>&
                           registerers_by_func) {
        std::string error;
        auto* wrapper = ::CalibrationWrapper::CreateWrapperCPPFromBuffer(
            data.ptr(), registerers_by_name, registerers_by_func, &error);
        if (!wrapper) {
          throw std::invalid_argument(error);  // throws ValueError in Python
        }
        return wrapper;
      }))
      .def("Prepare",
           [](CalibrationWrapper& self, py::handle& input_shapes,
              std::string signature_key) {
             return tensorflow::PyoOrThrow(
                 self.Prepare(input_shapes.ptr(), signature_key));
           })
      .def("Prepare",
           [](CalibrationWrapper& self, py::handle& input_shapes) {
             return tensorflow::PyoOrThrow(self.Prepare(input_shapes.ptr()));
           })
      .def("Prepare",
           [](CalibrationWrapper& self, std::string signature_key) {
             return tensorflow::PyoOrThrow(self.Prepare(signature_key));
           })
      .def("Prepare",
           [](CalibrationWrapper& self) {
             return tensorflow::PyoOrThrow(self.Prepare());
           })
      .def("FeedTensor",
           [](CalibrationWrapper& self, py::handle& input_value,
              std::string signature_key) {
             return tensorflow::PyoOrThrow(
                 self.FeedTensor(input_value.ptr(), signature_key));
           })
      .def("FeedTensor",
           [](CalibrationWrapper& self, py::handle& input_value) {
             return tensorflow::PyoOrThrow(self.FeedTensor(input_value.ptr()));
           })
      .def("QuantizeModel",
           [](CalibrationWrapper& self, int input_py_type, int output_py_type,
              bool allow_float, int activations_py_type, int bias_py_type,
              bool disable_per_channel,
              bool disable_per_channel_quantization_for_dense_layers) {
             return tensorflow::PyoOrThrow(self.QuantizeModel(
                 input_py_type, output_py_type, allow_float,
                 activations_py_type, bias_py_type, disable_per_channel,
                 disable_per_channel_quantization_for_dense_layers));
           })
      .def("QuantizeModel",
           [](CalibrationWrapper& self, int input_py_type, int output_py_type,
              bool allow_float, const char* operator_output_name) {
             return tensorflow::PyoOrThrow(
                 self.QuantizeModel(input_py_type, output_py_type, allow_float,
                                    operator_output_name));
           })
      .def("Calibrate", [](CalibrationWrapper& self) {
        return tensorflow::PyoOrThrow(self.Calibrate());
      });
}
