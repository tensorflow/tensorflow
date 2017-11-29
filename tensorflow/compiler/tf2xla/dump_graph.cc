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

#include "tensorflow/compiler/tf2xla/dump_graph_flags.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace dump_graph {

namespace {

struct NameCounts {
  mutex counts_mutex;
  std::unordered_map<string, int> counts;
};

string MakeUniquePath(string name) {
  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (int i = 0; i < name.size(); ++i) {
    if (name[i] == '/') {
      name[i] = '_';
    }
  }

  int count;
  {
    mutex_lock lock(instance.counts_mutex);
    count = instance.counts[name]++;
  }

  legacy_flags::DumpGraphFlags* flags = legacy_flags::GetDumpGraphFlags();
  string path = strings::StrCat(flags->tf_dump_graph_prefix, "/", name);
  if (count > 0) {
    strings::StrAppend(&path, "_", count);
  }
  strings::StrAppend(&path, ".pbtxt");
  return path;
}

}  // anonymous namespace

string DumpGraphDefToFile(const string& name, GraphDef const& graph_def) {
  string path = MakeUniquePath(name);
  TF_CHECK_OK(WriteTextProto(Env::Default(), path, graph_def));
  return path;
}

string DumpGraphToFile(const string& name, Graph const& graph,
                       const FunctionLibraryDefinition* flib_def) {
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  if (flib_def) {
    *graph_def.mutable_library() = flib_def->ToProto();
  }
  return DumpGraphDefToFile(name, graph_def);
}

string DumpFunctionDefToFile(const string& name, FunctionDef const& fdef) {
  string path = MakeUniquePath(name);
  TF_CHECK_OK(WriteTextProto(Env::Default(), path, fdef));
  return path;
}

}  // namespace dump_graph
}  // namespace tensorflow
