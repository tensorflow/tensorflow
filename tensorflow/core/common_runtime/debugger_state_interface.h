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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEBUGGER_STATE_INTERFACE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEBUGGER_STATE_INTERFACE_H_

#include <memory>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/debug.pb.h"

namespace tensorflow {

// Returns a summary string for the list of debug tensor watches.
const string SummarizeDebugTensorWatches(
    const protobuf::RepeatedPtrField<DebugTensorWatch>& watches);

// An abstract interface for storing and retrieving debugging information.
class DebuggerStateInterface {
 public:
  virtual ~DebuggerStateInterface() {}

  // Publish metadata about the debugged Session::Run() call.
  //
  // Args:
  //   global_step: A global step count supplied by the caller of
  //     Session::Run().
  //   session_run_index: A chronologically sorted index for calls to the Run()
  //     method of the Session object.
  //   executor_step_index: A chronologically sorted index of invocations of the
  //     executor charged to serve this Session::Run() call.
  //   input_names: Name of the input Tensors (feed keys).
  //   output_names: Names of the fetched Tensors.
  //   target_names: Names of the target nodes.
  virtual absl::Status PublishDebugMetadata(
      const int64_t global_step, const int64_t session_run_index,
      const int64_t executor_step_index, const std::vector<string>& input_names,
      const std::vector<string>& output_names,
      const std::vector<string>& target_nodes) = 0;
};

class DebugGraphDecoratorInterface {
 public:
  virtual ~DebugGraphDecoratorInterface() {}

  // Insert special-purpose debug nodes to graph and dump the graph for
  // record. See the documentation of DebugNodeInserter::InsertNodes() for
  // details.
  virtual absl::Status DecorateGraph(Graph* graph, Device* device) = 0;

  // Publish Graph to debug URLs.
  virtual absl::Status PublishGraph(const Graph& graph,
                                    const string& device_name) = 0;
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

  // If RegisterFactory() has been called, creates and supplies a concrete
  // DebuggerStateInterface implementation using the registered factory,
  // owned by the caller and return an OK Status. Otherwise returns an error
  // Status.
  static absl::Status CreateState(
      const DebugOptions& debug_options,
      std::unique_ptr<DebuggerStateInterface>* state);

 private:
  static DebuggerStateFactory* factory_;

  DebuggerStateRegistry(const DebuggerStateRegistry&) = delete;
  void operator=(const DebuggerStateRegistry&) = delete;
};

typedef std::function<std::unique_ptr<DebugGraphDecoratorInterface>(
    const DebugOptions& options)>
    DebugGraphDecoratorFactory;

class DebugGraphDecoratorRegistry {
 public:
  static void RegisterFactory(const DebugGraphDecoratorFactory& factory);

  static absl::Status CreateDecorator(
      const DebugOptions& options,
      std::unique_ptr<DebugGraphDecoratorInterface>* decorator);

 private:
  static DebugGraphDecoratorFactory* factory_;

  DebugGraphDecoratorRegistry(const DebugGraphDecoratorRegistry&) = delete;
  void operator=(const DebugGraphDecoratorRegistry&) = delete;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEBUGGER_STATE_INTERFACE_H_
