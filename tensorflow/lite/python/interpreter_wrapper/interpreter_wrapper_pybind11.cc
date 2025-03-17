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
        [](const std::string& model_path, int op_resolver_id,
           const std::vector<std::string>& registerers,
           bool preserve_all_tensors, bool disable_delegate_clustering) {
          std::string error;
          auto* wrapper = ::InterpreterWrapper::CreateWrapperCPPFromFile(
              model_path.c_str(), op_resolver_id, registerers, &error,
              preserve_all_tensors, disable_delegate_clustering);
          if (!wrapper) {
            throw std::invalid_argument(error);
          }
          return wrapper;
        });
  m.def(
      "CreateWrapperFromFile",
      [](const std::string& model_path, int op_resolver_id,
         const std::vector<std::string>& registerers_by_name,
         const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
         bool preserve_all_tensors, bool disable_delegate_clustering,
         int num_threads, bool default_delegate_latest_features) {
        std::string error;
        auto* wrapper = ::InterpreterWrapper::CreateWrapperCPPFromFile(
            model_path.c_str(), op_resolver_id, registerers_by_name,
            registerers_by_func, &error, preserve_all_tensors,
            disable_delegate_clustering, num_threads,
            default_delegate_latest_features);
        if (!wrapper) {
          throw std::invalid_argument(error);
        }
        return wrapper;
      });
  m.def("CreateWrapperFromBuffer",
        [](const py::bytes& data, int op_resolver_id,
           const std::vector<std::string>& registerers,
           bool preserve_all_tensors, bool disable_delegate_clustering) {
          std::string error;
          auto* wrapper = ::InterpreterWrapper::CreateWrapperCPPFromBuffer(
              data.ptr(), op_resolver_id, registerers, &error,
              preserve_all_tensors, disable_delegate_clustering);
          if (!wrapper) {
            throw std::invalid_argument(error);
          }
          return wrapper;
        });
  m.def(
      "CreateWrapperFromBuffer",
      [](const py::bytes& data, int op_resolver_id,
         const std::vector<std::string>& registerers_by_name,
         const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
         bool preserve_all_tensors, bool disable_delegate_clustering,
         int num_threads, bool default_delegate_latest_features) {
        std::string error;
        auto* wrapper = ::InterpreterWrapper::CreateWrapperCPPFromBuffer(
            data.ptr(), op_resolver_id, registerers_by_name,
            registerers_by_func, &error, preserve_all_tensors,
            disable_delegate_clustering, num_threads,
            default_delegate_latest_features);
        if (!wrapper) {
          throw std::invalid_argument(error);
        }
        return wrapper;
      });
  py::class_<InterpreterWrapper>(m, "InterpreterWrapper", py::module_local())
      .def(
          "AllocateTensors",
          [](InterpreterWrapper& self, int subgraph_index) {
            return tensorflow::PyoOrThrow(self.AllocateTensors(subgraph_index));
          },
          // LINT.IfChange
          py::arg("subgraph_index") = -1)
      // LINT.ThenChange(//tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc)
      .def(
          "Invoke",
          [](InterpreterWrapper& self, int subgraph_index) {
            return tensorflow::PyoOrThrow(self.Invoke(subgraph_index));
          },
          py::arg("subgraph_index") = 0)
      .def("InputIndices",
           [](const InterpreterWrapper& self) {
             return tensorflow::PyoOrThrow(self.InputIndices());
           })
      .def("OutputIndices",
           [](InterpreterWrapper& self) {
             return tensorflow::PyoOrThrow(self.OutputIndices());
           })
      .def(
          "ResizeInputTensor",
          [](InterpreterWrapper& self, int i, py::handle& value, bool strict,
             int subgraph_index) {
            return tensorflow::PyoOrThrow(
                self.ResizeInputTensor(i, value.ptr(), strict, subgraph_index));
          },
          py::arg("i"), py::arg("value"), py::arg("strict"),
          py::arg("subgraph_index") = 0)
      .def("NumTensors", &InterpreterWrapper::NumTensors)
      .def("NumSubgraphs", &InterpreterWrapper::NumSubgraphs)
      .def("TensorName", &InterpreterWrapper::TensorName)
      .def("TensorType",
           [](const InterpreterWrapper& self, int tensor_index,
              int subgraph_index) {
             return tensorflow::PyoOrThrow(
                 self.TensorType(tensor_index, subgraph_index));
           })
      .def("TensorSize",
           [](const InterpreterWrapper& self, int tensor_index,
              int subgraph_index) {
             return tensorflow::PyoOrThrow(
                 self.TensorSize(tensor_index, subgraph_index));
           })
      .def("TensorSizeSignature",
           [](const InterpreterWrapper& self, int tensor_index,
              int subgraph_index) {
             return tensorflow::PyoOrThrow(
                 self.TensorSizeSignature(tensor_index, subgraph_index));
           })
      .def("TensorSparsityParameters",
           [](const InterpreterWrapper& self, int tensor_index,
              int subgraph_index) {
             return tensorflow::PyoOrThrow(
                 self.TensorSparsityParameters(tensor_index, subgraph_index));
           })
      .def(
          "TensorQuantization",
          [](const InterpreterWrapper& self, int tensor_index,
             int subgraph_index) {
            return tensorflow::PyoOrThrow(
                self.TensorQuantization(tensor_index, subgraph_index));
          },
          R"pbdoc(
            Deprecated in favor of TensorQuantizationParameters.
          )pbdoc")
      .def("TensorQuantizationParameters",
           [](InterpreterWrapper& self, int tensor_index, int subgraph_index) {
             return tensorflow::PyoOrThrow(self.TensorQuantizationParameters(
                 tensor_index, subgraph_index));
           })
      .def(
          "SetTensor",
          [](InterpreterWrapper& self, int tensor_index, py::handle& value,
             int subgraph_index) {
            return tensorflow::PyoOrThrow(
                self.SetTensor(tensor_index, value.ptr(), subgraph_index));
          },
          py::arg("i"), py::arg("value"), py::arg("subgraph_index") = 0)
      .def(
          "GetTensor",
          [](const InterpreterWrapper& self, int tensor_index,
             int subgraph_index) {
            return tensorflow::PyoOrThrow(
                self.GetTensor(tensor_index, subgraph_index));
          },
          py::arg("tensor_index"), py::arg("subgraph_index") = 0)
      .def("GetSubgraphIndexFromSignature",
           [](InterpreterWrapper& self, const char* signature_key) {
             return tensorflow::PyoOrThrow(
                 self.GetSubgraphIndexFromSignature(signature_key));
           })
      .def("GetSignatureDefs",
           [](InterpreterWrapper& self) {
             return tensorflow::PyoOrThrow(self.GetSignatureDefs());
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
          [](InterpreterWrapper& self, py::handle& base_object,
             int tensor_index, int subgraph_index) {
            return tensorflow::PyoOrThrow(
                self.tensor(base_object.ptr(), tensor_index, subgraph_index));
          },
          R"pbdoc(
            Returns a reference to tensor index as a numpy array from subgraph.
            The base_object should be the interpreter object providing the
            memory.
          )pbdoc",
          py::arg("base_object"), py::arg("tensor_index"),
          py::arg("subgraph_index") = 0)
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
          )pbdoc")
      .def("interpreter", [](InterpreterWrapper& self) {
        return reinterpret_cast<intptr_t>(self.interpreter());
      });
}
