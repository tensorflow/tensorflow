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
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

DebugDataDumper* DebugDataDumper::Global() {
  static DebugDataDumper* global_instance_ = new DebugDataDumper();
  return global_instance_;
}

bool DebugDataDumper::ShouldDump(const std::string& name) const {
  // Do not dump data for the wrapped functions.
  if (absl::StartsWith(name, "__wrapped__")) return false;

  // Get the name filter from TF_DUMP_GRAPH_NAME_FILTER.
  const char* name_filter = getenv("TF_DUMP_GRAPH_NAME_FILTER");
  if (name_filter == nullptr) {
    VLOG(1) << "Skip dumping graph '" << name
            << "', because TF_DUMP_GRAPH_NAME_FILTER is not set";
    return false;
  }

  // If name_filter is not '*' or name doesn't contain the name_filter,
  // skip the dump.
  std::string str_name_filter = std::string(name_filter);
  if (str_name_filter != "*" &&
      name.find(str_name_filter) == std::string::npos) {
    VLOG(1) << "Skip dumping graph '" << name
            << "', because TF_DUMP_GRAPH_NAME_FILTER is not '*' and "
            << "it is not contained by the graph name";
    return false;
  }

  // If all conditions are met, return true to allow the dump.
  return true;
}

void DebugDataDumper::DumpOpCreationStackTraces(const std::string& name,
                                                const std::string& tag,
                                                const Graph* graph) {
  const char* dump_stacktraces = getenv("TF_DUMP_OP_CREATION_STACKTRACES");
  if (dump_stacktraces == nullptr) {
    VLOG(1) << "Skip dumping op creation stacktraces for '" << name
            << "', because TF_DUMP_OP_CREATION_STACKTRACES is not set";
    return;
  }

  // Construct the dump filename.
  std::string dump_filename = GetDumpFilename(name, tag);

  // Dump module txt to file.
  DumpToFile(dump_filename, "", ".csv", "StackTrace",
             [&graph, &dump_filename](WritableFile* file) {
               auto status = file->Append("node_id,node_name,stackframes\n");
               if (!status.ok()) {
                 LOG(WARNING) << "error writing to file to " << dump_filename
                              << ": " << status.error_message();
                 return status;
               }

               for (Node* node : graph->nodes()) {
                 auto stack_trace = node->GetStackTrace();
                 if (stack_trace == nullptr) continue;

                 int node_id = node->id();
                 std::string node_name = node->name();
                 std::vector<std::string> stackframes;

                 for (auto& frame : stack_trace->ToFrames()) {
                   stackframes.push_back(
                       absl::StrFormat("%s(%d): %s", frame.file_name,
                                       frame.line_number, frame.function_name));
                 }

                 status = file->Append(
                     absl::StrFormat("%d,%s,%s\n", node_id, node_name,
                                     absl::StrJoin(stackframes, ";")));

                 if (!status.ok()) {
                   LOG(WARNING) << "error writing to file to " << dump_filename
                                << ": " << status.error_message();
                   return status;
                 }
               }

               return file->Close();
             });
}

void DebugDataDumper::DumpGraph(const std::string& name, const std::string& tag,
                                const Graph* graph,
                                const FunctionLibraryDefinition* func_lib_def) {
  // Construct the dump filename.
  std::string dump_filename = GetDumpFilename(name, tag);

  // Make sure the dump filename is not longer than 255,
  // because Linux won't take filename that long.
  if (dump_filename.size() > 255) {
    LOG(WARNING) << "Failed to dump graph " << dump_filename << " to "
                 << ", because the file name is longer than 255";
    return;
  }

  // Construct a graph def.
  GraphDef graph_def;
  graph->ToGraphDef(&graph_def);

  if (func_lib_def) *graph_def.mutable_library() = func_lib_def->ToProto();

  // Now dump the graph into the target file.
  DumpGraphDefToFile(dump_filename, graph_def);
}

std::string DebugDataDumper::GetDumpFilename(const std::string& name,
                                             const std::string& tag) {
  std::string dump_name = name.empty() ? "unknown_graph" : name;
  return absl::StrFormat("%s.%04d.%s", dump_name, GetNextDumpId(name), tag);
}

}  // namespace tensorflow
