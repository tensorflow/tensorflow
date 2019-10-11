/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_DUMP_GRAPH_H_
#define TENSORFLOW_CORE_UTIL_DUMP_GRAPH_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Dumps 'graph_def' to a file, as a GraphDef text proto. Returns the file name
// chosen.
//
// If the TF_DUMP_GRAPH_PREFIX environment variable is "-", then instead the
// GraphDef will be logged (using the LOG() macro).
//
// Automatically picks a file name. Prefixes 'name' with the value of the
// TF_DUMP_GRAPH_PREFIX environment variable if 'dirname' is empty, and suffixes
// 'name' with ".pbtxt" to form a name. If a graph has already been dumped by
// this process with the same name, suffixes with "_n.pbtxt", where 'n' is a
// sequence number.
string DumpGraphDefToFile(const string& name, GraphDef const& graph_def,
                          const string& dirname = "");

// Similar to DumpGraphDefToFile, but builds the GraphDef to dump from a 'graph'
// and an optional function library 'flib_def'. Returns the file name chosen.
string DumpGraphToFile(const string& name, Graph const& graph,
                       const FunctionLibraryDefinition* flib_def = nullptr,
                       const string& dirname = "");

// Similar to DumpGraphDefToFile, but dumps a function as a FunctionDef text
// proto. Returns the file name chosen.
string DumpFunctionDefToFile(const string& name, FunctionDef const& fdef,
                             const string& dirname = "");

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_DUMP_GRAPH_H_
