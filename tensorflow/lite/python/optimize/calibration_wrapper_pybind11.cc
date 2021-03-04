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
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
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
      .def(py::init([](py::handle& data) {
        auto* wrapper =
            ::CalibrationWrapper::CreateWrapperCPPFromBuffer(data.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return wrapper;
      }))
      .def("Prepare",
           [](CalibrationWrapper& self, py::handle& input_shapes) {
             return tensorflow::PyoOrThrow(self.Prepare(input_shapes.ptr()));
           })
      .def("Prepare",
           [](CalibrationWrapper& self) {
             return tensorflow::PyoOrThrow(self.Prepare());
           })
      .def("FeedTensor",
           [](CalibrationWrapper& self, py::handle& input_value) {
             return tensorflow::PyoOrThrow(self.FeedTensor(input_value.ptr()));
           })
      .def("QuantizeModel",
           [](CalibrationWrapper& self, int input_py_type, int output_py_type,
              bool allow_float, int activations_py_type,
              bool enable_mlir_quantizer) {
             return tensorflow::PyoOrThrow(
                 self.QuantizeModel(input_py_type, output_py_type, allow_float,
                                    activations_py_type));
           })
      .def("QuantizeModel",
           [](CalibrationWrapper& self, int input_py_type, int output_py_type,
              bool allow_float, int activations_py_type) {
             return tensorflow::PyoOrThrow(
                 self.QuantizeModel(input_py_type, output_py_type, allow_float,
                                    activations_py_type));
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
