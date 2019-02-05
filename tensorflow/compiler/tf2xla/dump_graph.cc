/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Helper functions for dumping Graphs, GraphDefs, and FunctionDefs to files for
// debugging.

#include "tensorflow/compiler/tf2xla/dump_graph.h"

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace dump_graph {

string DumpGraphDefToFile(const string& name, GraphDef const& graph_def) {
  return tensorflow::DumpGraphDefToFile(
      name, graph_def, GetDumpGraphFlags()->tf_dump_graph_prefix);
}

string DumpGraphToFile(const string& name, Graph const& graph,
                       const FunctionLibraryDefinition* flib_def) {
  return tensorflow::DumpGraphToFile(name, graph, flib_def,
                                     GetDumpGraphFlags()->tf_dump_graph_prefix);
}

string DumpFunctionDefToFile(const string& name, FunctionDef const& fdef) {
  return tensorflow::DumpFunctionDefToFile(
      name, fdef, GetDumpGraphFlags()->tf_dump_graph_prefix);
}

}  // namespace dump_graph
}  // namespace tensorflow
