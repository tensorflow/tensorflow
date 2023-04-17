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

#ifndef TENSORFLOW_CORE_UTIL_DEBUG_DATA_DUMPER_H_
#define TENSORFLOW_CORE_UTIL_DEBUG_DATA_DUMPER_H_

#include <optional>
#include <set>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/mutex.h"

#define DEBUG_DATA_DUMPER() ::tensorflow::DebugDataDumper::Global()

inline constexpr const char* kDebugGroupMain = "main";
inline constexpr const char* kDebugGroupOpStacktrace = "op_stacktrace";
inline constexpr const char* kDebugGroupGraphOptPass = "graph_opt_pass";
inline constexpr const char* kDebugGroupBridgePhase1 = "bridge_phase1";
inline constexpr const char* kDebugGroupBridgePhase2 = "bridge_phase2";

namespace tensorflow {

class FunctionLibraryDefinition;
class Graph;

////////////////////////////////////////////////////////////////////////////////
// This class is responsible for dumping debugging data (e.g., GraphDef, MLIR).
//
// To dump GraphDef/MLIRs, take the following steps:
// * Set envvar TF_DUMP_GRAPH_PREFIX to your target dump directory.
// * Set envvar TF_DUMP_GRAPH_NAME_FILTER to '*' to dump all graphs,
//   or a name filter to dump graphs with a name containing it.
// * Set envvar TF_DUMP_GRAPH_GROUPS to your dump groups (comma-separated).
//
// The dumped graphs then can be found in your target dump directory.
// The filename of the dump looks like this:
// <name>.<order-id>.<group>.<tag>
//
// This is what each field means:
// * <name>     : The name of your dump.
// * <order-id> : The order of dumps of a specific name.
//                Lower orders are executed before higher orders.
// * <group>    : The group of your dump, e.g., main.
// * <tag>      : The tag of your dump, e.g., your pass name.
//
// Example dump files are:
// __inference_train_step_441.0.main.before_pre_placement_passes.pbtxt
// __inference_train_step_441.1.main.before_placer.pbtxt
// __inference_train_step_441.2.main.before_post_placement_passes.pbtxt
// __inference_train_step_441.3.main.before_graph_optimization.pbtxt
// __inference_train_step_441.4.main.after_graph_optimization.pbtxt
// __inference_train_step_441.5.main.before_post_rewrite_for_exec_passes.pbtxt
////////////////////////////////////////////////////////////////////////////////
class DebugDataDumper {
 public:
  // Get the singleton instance.
  static DebugDataDumper* Global();

  // Initialize the debug data dumper.
  void LoadEnvvars();

  // Check if we should dump debug data.
  // We should dump debug data only if the followings are true:
  // 1. Envvar TF_DUMP_GRAPH_PREFIX is set to your target dump directory.
  // 2. This condition is true if one of the followings is true.
  //    2.1. TF_DUMP_GRAPH_NAME_FILTER is set to '*'
  //    2.2. TF_DUMP_GRAPH_NAME_FILTER is set to a name filter
  //         which is a substr of name.
  // 3. The group is defined in TF_DUMP_GRAPH_GROUPS.
  bool ShouldDump(const std::string& name, const std::string& group) const;

  // Dump op creation callstacks, if ShouldDump returns true.
  void DumpOpCreationStackTraces(const std::string& name,
                                 const std::string& group,
                                 const std::string& tag, const Graph* graph);

  // Dump a graph, if ShouldDump returns true.
  void DumpGraph(const std::string& name, const std::string& group,
                 const std::string& tag, const Graph* graph,
                 const FunctionLibraryDefinition* func_lib_def,
                 bool bypass_filter = false);

  // Get the dump file basename. Dump file basenames are in this format:
  // <name>.<order-id>.<group>.<tag>
  //
  // What each field means is explained on the class level comment.
  std::string GetDumpFilename(const std::string& name, const std::string& group,
                              const std::string& tag);

 private:
  DebugDataDumper();

  // Get next dump id for a name.
  int GetNextDumpId(const std::string& name) {
    // Use a lock to make sure this is thread safe.
    const mutex_lock lock(lock_);
    return dump_order_ids_[name]++;
  }

  // A dict to maintain the mapping from dump name to its current dump id.
  absl::flat_hash_map<std::string, int> dump_order_ids_;

  // A mutex to make sure this is thread safe.
  tensorflow::mutex lock_;

  // The name filter.
  std::optional<std::string> name_filter_;

  // The groups filter.
  std::set<string> groups_filter_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_DEBUG_DATA_DUMPER_H_
