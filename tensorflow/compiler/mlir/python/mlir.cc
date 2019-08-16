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

#include "llvm/Support/raw_ostream.h"
#include "include/pybind11/pybind11.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"

namespace tensorflow {

namespace py = pybind11;

namespace {

// Simple wrapper to support tf.mlir.experimental.convert_graph_def.
// Load a .pbptx, convert to MLIR, and (optionally) optimize the module before
// returning it as a string.
// This is an early experimental API, ideally we should return a wrapper object
// around a Python binding to the MLIR module.
std::string import_graphdef(const std::string &proto) {
  GraphDef graphdef;
  if (!tensorflow::LoadProtoFromBuffer(proto, &graphdef).ok()) {
    throw std::runtime_error("Error parsing proto, see logs for error.");
  }
  GraphDebugInfo debug_info;
  NodeSpecs specs;
  mlir::MLIRContext context;
  auto module = ConvertGraphdefToMlir(graphdef, debug_info, specs, &context);
  if (!module.ok()) {
    throw std::runtime_error(module.status().error_message());
  }

  std::string txt_module;
  {
    llvm::raw_string_ostream os{txt_module};
    module.ConsumeValueOrDie()->print(os);
  }
  return txt_module;
}

}  // namespace

PYBIND11_MODULE(mlir_extension, m) {
  m.def("import_graphdef", import_graphdef,
        "Import textual graphdef and return a textual MLIR module.");
}

}  // namespace tensorflow
