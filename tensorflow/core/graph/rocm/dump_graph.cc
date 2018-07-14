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

#ifdef TENSORFLOW_USE_ROCM

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "dump_graph.h"

namespace tensorflow {
namespace rtglib {    
namespace dump_graph {

struct NameCounts {
  mutex counts_mutex;
  std::unordered_map<string, int> counts;
};

string MakeUniquePath(string name) {
  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (int i = 0; i < name.size(); ++i) {
      if (name[i] == '/' || std::isspace(name[i])) {
        name[i] = '_';
      }
  }

  int count;
  {
    mutex_lock lock(instance.counts_mutex);
    count = instance.counts[name]++;
  }

  string path = strings::StrCat("./", name);
  if (count > 0) {
    strings::StrAppend(&path, "_", count);
  }
  strings::StrAppend(&path, ".pbtxt");
  return path;
}

    
void DumpGraphDefToFile(const string& name, const GraphDef& graph_def) {
  string path = MakeUniquePath(name);
  TF_CHECK_OK(WriteTextProto(Env::Default(), path, graph_def));
}

void DumpGraphToFile(const string& name, const Graph& graph) {
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  return DumpGraphDefToFile(name, graph_def);
}
    
} // namespace dump_graph
} // namespace rtglib
} // namespace tensorflow


#endif // TENSORFLOW_USE_ROCM
