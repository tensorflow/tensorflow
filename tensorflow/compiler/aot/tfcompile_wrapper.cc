/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "include/pybind11/cast.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"
#include "tensorflow/compiler/aot/compile.h"
#include "tensorflow/compiler/aot/flags.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_tfcompile, m) {
  m.doc() = R"pbdoc(
    _pywrap_tfcompile
    -----
  )pbdoc";

  m.def(
      "Compile",
      [](std::string graph, std::string config, std::string target_triple,
         std::string target_cpu, std::string target_features,
         std::string entry_point, std::string cpp_class,
         std::string out_function_object, std::string out_metadata_object,
         std::string out_header, std::string out_session_module,
         std::string mlir_components, std::string tensorflow_header_root,
         bool gen_name_to_index, bool gen_program_shape) {
        tensorflow::tfcompile::MainFlags flags;
        flags.graph = std::move(graph);
        flags.config = std::move(config);
        flags.target_triple = std::move(target_triple);
        flags.target_cpu = std::move(target_cpu);
        flags.target_features = std::move(target_features);
        flags.entry_point = std::move(entry_point);
        flags.cpp_class = std::move(cpp_class);
        flags.out_function_object = std::move(out_function_object);
        flags.out_metadata_object = std::move(out_metadata_object);
        flags.out_header = std::move(out_header);
        flags.out_session_module = std::move(out_session_module);
        flags.mlir_components = std::move(mlir_components);
        flags.tensorflow_header_root = std::move(tensorflow_header_root);

        // C++ codegen options
        flags.gen_name_to_index = gen_name_to_index;
        flags.gen_program_shape = gen_program_shape;

        tensorflow::MaybeRaiseFromStatus(tensorflow::tfcompile::Main(flags));
      },
      py::arg("graph") = "", py::arg("config") = "",
      py::arg("target_triple") = "x86_64-pc-linux", py::arg("target_cpu") = "",
      py::arg("target_features") = "", py::arg("entry_point") = "entry",
      py::arg("cpp_class") = "", py::arg("out_function_object") = "out_model.o",
      py::arg("out_metadata_object") = "out_helper.o",
      py::arg("out_header") = "out.h", py::arg("out_session_module") = "",
      py::arg("mlir_components") = "",
      py::arg("tensorflow_header_root") = "third_party/tensorflow",
      py::arg("gen_name_to_index") = false,
      py::arg("gen_program_shape") = false);
}
