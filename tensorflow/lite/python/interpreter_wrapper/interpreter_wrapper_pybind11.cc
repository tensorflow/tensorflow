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
#include "pybind11/stl.h"
#include "tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace py = pybind11;
using tflite::interpreter_wrapper::InterpreterWrapper;

PYBIND11_MODULE(_pywrap_tensorflow_interpreter_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_tensorflow_interpreter_wrapper
    -----
  )pbdoc";

  // pybind11 suggests to convert factory functions into constructors, but
  // when bytes are provided the wrapper will be confused which
  // constructor to call.
  m.def("CreateWrapperFromFile",
        [](const std::string& model_path,
           const std::vector<std::string>& registerers) {
          std::string error;
          auto* wrapper = ::InterpreterWrapper::CreateWrapperCPPFromFile(
              model_path.c_str(), registerers, &error);
          if (!wrapper) {
            throw std::invalid_argument(error);
          }
          return wrapper;
        });
  m.def("CreateWrapperFromBuffer",
        [](const py::bytes& data, const std::vector<std::string>& registerers) {
          std::string error;
          auto* wrapper = ::InterpreterWrapper::CreateWrapperCPPFromBuffer(
              data.ptr(), registerers, &error);
          if (!wrapper) {
            throw std::invalid_argument(error);
          }
          return wrapper;
        });
  py::class_<InterpreterWrapper>(m, "InterpreterWrapper")
      .def("AllocateTensors",
           [](InterpreterWrapper& self) {
             return tensorflow::PyoOrThrow(self.AllocateTensors());
           })
      .def("Invoke",
           [](InterpreterWrapper& self) {
             return tensorflow::PyoOrThrow(self.Invoke());
           })
      .def("InputIndices",
           [](const InterpreterWrapper& self) {
             return tensorflow::PyoOrThrow(self.InputIndices());
           })
      .def("OutputIndices",
           [](InterpreterWrapper& self) {
             return tensorflow::PyoOrThrow(self.OutputIndices());
           })
      .def("ResizeInputTensor",
           [](InterpreterWrapper& self, int i, py::handle& value, bool strict) {
             return tensorflow::PyoOrThrow(
                 self.ResizeInputTensor(i, value.ptr(), strict));
           })
      .def("NumTensors", &InterpreterWrapper::NumTensors)
      .def("TensorName", &InterpreterWrapper::TensorName)
      .def("TensorType",
           [](const InterpreterWrapper& self, int i) {
             return tensorflow::PyoOrThrow(self.TensorType(i));
           })
      .def("TensorSize",
           [](const InterpreterWrapper& self, int i) {
             return tensorflow::PyoOrThrow(self.TensorSize(i));
           })
      .def("TensorSizeSignature",
           [](const InterpreterWrapper& self, int i) {
             return tensorflow::PyoOrThrow(self.TensorSizeSignature(i));
           })
      .def("TensorSparsityParameters",
           [](const InterpreterWrapper& self, int i) {
             return tensorflow::PyoOrThrow(self.TensorSparsityParameters(i));
           })
      .def(
          "TensorQuantization",
          [](const InterpreterWrapper& self, int i) {
            return tensorflow::PyoOrThrow(self.TensorQuantization(i));
          },
          R"pbdoc(
            Deprecated in favor of TensorQuantizationParameters.
          )pbdoc")
      .def(
          "TensorQuantizationParameters",
          [](InterpreterWrapper& self, int i) {
            return tensorflow::PyoOrThrow(self.TensorQuantizationParameters(i));
          })
      .def("SetTensor",
           [](InterpreterWrapper& self, int i, py::handle& value) {
             return tensorflow::PyoOrThrow(self.SetTensor(i, value.ptr()));
           })
      .def("GetTensor",
           [](const InterpreterWrapper& self, int i) {
             return tensorflow::PyoOrThrow(self.GetTensor(i));
           })
      .def("ResetVariableTensors",
           [](InterpreterWrapper& self) {
             return tensorflow::PyoOrThrow(self.ResetVariableTensors());
           })
      .def("NumNodes", &InterpreterWrapper::NumNodes)
      .def("NodeName", &InterpreterWrapper::NodeName)
      .def("NodeInputs",
           [](const InterpreterWrapper& self, int i) {
             return tensorflow::PyoOrThrow(self.NodeInputs(i));
           })
      .def("NodeOutputs",
           [](const InterpreterWrapper& self, int i) {
             return tensorflow::PyoOrThrow(self.NodeOutputs(i));
           })
      .def(
          "tensor",
          [](InterpreterWrapper& self, py::handle& base_object, int i) {
            return tensorflow::PyoOrThrow(self.tensor(base_object.ptr(), i));
          },
          R"pbdoc(
            Returns a reference to tensor index i as a numpy array. The
            base_object should be the interpreter object providing the memory.
          )pbdoc")
      .def(
          "ModifyGraphWithDelegate",
          // Address of the delegate is passed as an argument.
          [](InterpreterWrapper& self, uintptr_t delegate_ptr) {
            return tensorflow::PyoOrThrow(self.ModifyGraphWithDelegate(
                reinterpret_cast<TfLiteDelegate*>(delegate_ptr)));
          },
          R"pbdoc(
            Adds a delegate to the interpreter.
          )pbdoc")
      .def(
          "SetNumThreads",
          [](InterpreterWrapper& self, int num_threads) {
            return tensorflow::PyoOrThrow(self.SetNumThreads(num_threads));
          },
          R"pbdoc(
             ask the interpreter to set the number of threads to use.
          )pbdoc");
}
