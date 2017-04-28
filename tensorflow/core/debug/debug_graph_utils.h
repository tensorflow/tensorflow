/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DEBUG_NODE_INSERTER_H_
#define TENSORFLOW_DEBUG_NODE_INSERTER_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/debug.pb.h"

namespace tensorflow {

class DebuggerState : public DebuggerStateInterface {
 public:
  DebuggerState(const DebugOptions& debug_options);
  virtual ~DebuggerState();

  // Returns a summary string for RepeatedPtrFields of DebugTensorWatches.
  const string SummarizeDebugTensorWatches() override;

  // Insert special-purpose debug nodes to graph. See the documentation of
  // DebugNodeInserter::InsertNodes() for details.
  Status DecorateGraphForDebug(Graph* graph, Device* device) override;

  const protobuf::RepeatedPtrField<DebugTensorWatch>& watches;

  // Publish metadata about the debugged Session::Run() call.
  //
  // See the doc string of DebuggerStateInterface::PublishDebugMetadata() for
  // details.
  Status PublishDebugMetadata(const int64 global_step,
                              const int64 session_run_count,
                              const int64 executor_step_count,
                              const std::vector<string>& input_names,
                              const std::vector<string>& output_names,
                              const std::vector<string>& target_names) override;

 private:
  std::unordered_set<string> debug_urls_;
};

class DebugNodeInserter {
 public:
  // EXPERIMENTAL: Insert special debug ops (e.g., DebugIdentity) to graph for
  // debugging. Currently, such ops need to take exactly one input and has the
  // string attribute "tensor_name" to indicate what tensor it watches.
  // For example, before the node insertion, the graph may look like:
  //
  // A:0 -----------1----------> B
  //      |
  //      ---------2-----------> C
  //
  // wherein the output slot 0 of node A feeds as the input to nodes B through
  // edge 1 and to node C through edge 2.
  // After the node insertion, assuming both B and C have non-Ref input, the
  // graph becomes:
  // A:0 ---3---> Copy -----------4----------> B
  //                       |
  //                       ---------5--------> C
  //                       |
  //                       ---------6--------> X
  //
  // If a node (e.g., B) has Ref input, the graph becomes:
  //
  //           --------------------------------> B
  //           |
  // A:0 ---3-----> Copy -----------4----------> C
  //                       |
  //                       -----------5--------> X
  //
  // In other words, we do not feed Refs to deep-copies to downstream nodes.
  //
  // Copy is the inserted deep-copy node that copies the input tensor on-device
  // (e.g., CPU-to-CPU or GPU-to-GPU deep copy) that reduces the likelihood of
  // racy updates during the debug watches. X is the newly created debug node
  // that transforms the input (copy of the watched tensor) into a debug signal.
  //
  // DebugIdentity is the simplest debugging paradigm, in which the debug signal
  // (i.e., X:0) equals the tensor itself. More sophisticated debug ops can be
  // used to transform the tensor into other debug signals. An example is the
  // DebugNanCounter op.
  //
  // If the nodes (A, B and C) are located on GPU and the edges from A to B or C
  // is HOST_MEMORY, then the CopyHost op will be used instead of the Copy op.
  static Status InsertNodes(
      const protobuf::RepeatedPtrField<DebugTensorWatch>& watches, Graph* graph,
      Device* device);

  // Set the parallel_iterations attribute of TensorFlow while loops
  // (specifically the nodes for which IsEnter() returns true) to 1 to prevent
  // any node from being executed multiple times concurrently and
  // generating temporally-overlapping debug Tensor dumps.
  static void DeparallelizeWhileLoops(Graph* graph, Device* device);

  // Get canonical name of a copy node.
  static const string GetCopyNodeName(const string& node_name,
                                      const int output_slot);

  // Get canonical name of a debug node.
  static const string GetDebugNodeName(const string& tensor_name,
                                       const int debug_op_num,
                                       const string& debug_op_name);

 private:
  static Status CreateCopyNode(
      Graph* graph, const DeviceType device_type, const bool is_host_memory,
      const string& src_node_name, const int src_output, const DataType src_dt,
      const string& tensor_name, const std::vector<string>& debug_ops,
      const std::vector<string>& debug_urls, Node** copy_node);

  // Parse the debug_op_name string to extract proper op name and attributes.
  // debug_op_name can be the proper op name only, e.g., "DebugNumericSummary".
  // It can also contain customizable keys and values. Each key-value pair is
  // connected with an equal sign ("="). Multiple key-value pairs are separated
  // with semicolons (";"), which optional whitespace in between, e.g.,
  // "DebugNumericSummary(mute_if_healthy=true, lower_bound=-100.0)".
  static Status ParseDebugOpName(
      const string& debug_op_name, string* debug_op_name_proper,
      std::unordered_map<string, string>* attributes);

  static Status SetDebugNodeAttributes(
      Node* debug_node, const std::unordered_map<string, string>& attributes);

  static Status CreateDebugNode(Graph* graph, const DeviceType device_type,
                                const string& src_copy_node_name,
                                const DataType src_dt,
                                const string& tensor_name,
                                const std::vector<string>& debug_urls,
                                const int debug_op_num,
                                const string& debug_op_name, Node** debug_node);

  friend class DebugGraphUtilsTest;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_DEBUG_NODE_INSERTER_H_
