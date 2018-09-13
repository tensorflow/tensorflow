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

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/dump_graph_flags.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace dump_graph {

namespace {

struct NameCounts {
  mutex counts_mutex;
  std::unordered_map<string, int> counts;
};

string MakeUniqueFilename(string name) {
  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (int i = 0; i < name.size(); ++i) {
    char ch = name[i];
    if (ch == '/' || ch == '[' || ch == ']' || ch == '*' || ch == '?') {
      name[i] = '_';
    }
  }

  int count;
  {
    mutex_lock lock(instance.counts_mutex);
    count = instance.counts[name]++;
  }

  string filename = name;
  if (count > 0) {
    absl::StrAppend(&filename, "_", count);
  }
  absl::StrAppend(&filename, ".pbtxt");
  return filename;
}

string WriteTextProtoToUniqueFile(
    Env* env, const string& name, const char* proto_type,
    const ::tensorflow::protobuf::Message& proto) {
  const string& dirname =
      legacy_flags::GetDumpGraphFlags()->tf_dump_graph_prefix;
  Status status = env->RecursivelyCreateDir(dirname);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to create " << dirname << " for dumping "
                 << proto_type << ": " << status;
    return "(unavailable)";
  }
  string filepath = absl::StrCat(dirname, "/", MakeUniqueFilename(name));
  status = WriteTextProto(Env::Default(), filepath, proto);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to dump " << proto_type << " to file: " << filepath
                 << " : " << status;
    return "(unavailable)";
  }
  LOG(INFO) << "Dumped " << proto_type << " to " << filepath;
  return filepath;
}

}  // anonymous namespace

string DumpGraphDefToFile(const string& name, GraphDef const& graph_def) {
  return WriteTextProtoToUniqueFile(Env::Default(), name, "GraphDef",
                                    graph_def);
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
  return WriteTextProtoToUniqueFile(Env::Default(), name, "FunctionDef", fdef);
}

}  // namespace dump_graph
}  // namespace tensorflow
