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

#ifndef TENSORFLOW_COMMON_RUNTIME_DEBUGGER_STATE_INTERFACE_H_
#define TENSORFLOW_COMMON_RUNTIME_DEBUGGER_STATE_INTERFACE_H_

#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class DebugOptions;  // Defined in core/protobuf/debug.h.
class Device;
class Graph;

// An abstract interface for storing and retrieving debugging information.
class DebuggerStateInterface {
 public:
  virtual ~DebuggerStateInterface() {}

  // Returns a summary string for RepeatedPtrFields of DebugTensorWatches.
  virtual const string SummarizeDebugTensorWatches() = 0;

  // Insert special-purpose debug nodes to graph and dump the graph for
  // record. See the documentation of DebugNodeInserter::InsertNodes() for
  // details.
  virtual Status DecorateGraphForDebug(Graph* graph, Device* device) = 0;

  // Publish metadata about the debugged Session::Run() call.
  //
  // Args:
  //   global_step: A global step count supplied by the caller of
  //     Session::Run().
  //   session_run_count: A counter for calls to the Run() method of the
  //     Session object.
  //   executor_step_count: A counter for invocations of the executor charged
  //     to serve this Session::Run() call.
  //   input_names: Name of the input Tensors (feed keys).
  //   output_names: Names of the fetched Tensors.
  //   target_names: Names of the target nodes.
  virtual Status PublishDebugMetadata(
      const int64 global_step, const int64 session_run_count,
      const int64 executor_step_count, const std::vector<string>& input_names,
      const std::vector<string>& output_names,
      const std::vector<string>& target_nodes) = 0;
};

typedef std::function<std::unique_ptr<DebuggerStateInterface>(
    const DebugOptions& options)>
    DebuggerStateFactory;

// Contains only static methods for registering DebuggerStateFactory.
// We don't expect to create any instances of this class.
// Call DebuggerStateRegistry::RegisterFactory() at initialization time to
// define a global factory that creates instances of DebuggerState, then call
// DebuggerStateRegistry::CreateState() to create a single instance.
class DebuggerStateRegistry {
 public:
  // Registers a function that creates a concrete DebuggerStateInterface
  // implementation based on DebugOptions.
  static void RegisterFactory(const DebuggerStateFactory& factory);

  // If RegisterFactory() has been called, creates and returns a concrete
  // DebuggerStateInterface implementation using the registered factory,
  // owned by the caller.  Otherwise returns nullptr.
  static std::unique_ptr<DebuggerStateInterface> CreateState(
      const DebugOptions& debug_options);

 private:
  static DebuggerStateFactory* factory_;

  TF_DISALLOW_COPY_AND_ASSIGN(DebuggerStateRegistry);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DEBUGGER_STATE_INTERFACE_H_
