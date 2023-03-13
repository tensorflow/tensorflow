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

#include "tensorflow/core/util/debug_data_dumper.h"

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

DebugDataDumper* DebugDataDumper::Global() {
  static DebugDataDumper* global_instance_ = new DebugDataDumper();
  return global_instance_;
}

void DebugDataDumper::DumpGraph(const std::string& name, const Graph* graph,
                                const std::string& tag) {
  // Use a lock to make sure this is thread safe.
  const mutex_lock lock(lock_);

  // Get the directory name from TF_DUMP_GRAPH_PREFIX.
  const char* dump_prefix = getenv("TF_DUMP_GRAPH_PREFIX");

  // Get the name filter from TF_DUMP_GRAPH_NAME_FILTER.
  const char* name_filter = getenv("TF_DUMP_GRAPH_NAME_FILTER");
  if (name_filter == nullptr) {
    VLOG(1) << "Skip dumping graph '" << name
            << "', because TF_DUMP_GRAPH_NAME_FILTER is not set";
    return;
  }

  // If name_filter is not '*' or name doesn't contain the name_filter,
  // skip the dump.
  std::string str_name_filter = std::string(name_filter);
  if (str_name_filter != "*" &&
      name.find(str_name_filter) == std::string::npos) {
    VLOG(1) << "Skip dumping graph '" << name
            << "', because TF_DUMP_GRAPH_NAME_FILTER is not '*' and "
            << "it is not contained by the graph name";
    return;
  }

  // Construct the dump filename.
  std::string dump_filename =
      absl::StrFormat("%s.%d.%s", name, GetNextDumpId(name), tag);

  // Make sure the dump filename is not longer than 255,
  // because Linux won't take filename that long.
  if (dump_filename.size() > 255) {
    LOG(WARNING) << "Failed to dump graph " << dump_filename << " to "
                 << dump_prefix << ", because the file name is longer than 255";
    return;
  }

  // Now dump the graph into the target file.
  LOG(INFO) << "Dumping graph " << dump_filename << " to " << dump_prefix;
  DumpGraphToFile(dump_filename, *graph, nullptr,
                  dump_prefix ? std::string(dump_prefix) : "");
}

}  // namespace tensorflow
