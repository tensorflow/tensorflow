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

#include "tensorflow/core/debug/debugger_state_impl.h"

#include "tensorflow/core/debug/debug_graph_utils.h"
#include "tensorflow/core/debug/debug_io_utils.h"

namespace tensorflow {

DebuggerState::DebuggerState(const DebugOptions& debug_options) {
  for (const DebugTensorWatch& watch :
       debug_options.debug_tensor_watch_opts()) {
    for (const string& url : watch.debug_urls()) {
      debug_urls_.insert(url);
    }
  }
}

DebuggerState::~DebuggerState() {
  for (const string& debug_url : debug_urls_) {
    DebugIO::CloseDebugURL(debug_url).IgnoreError();
  }
}

Status DebuggerState::PublishDebugMetadata(
    const int64 global_step, const int64 session_run_index,
    const int64 executor_step_index, const std::vector<string>& input_names,
    const std::vector<string>& output_names,
    const std::vector<string>& target_names) {
  return DebugIO::PublishDebugMetadata(global_step, session_run_index,
                                       executor_step_index, input_names,
                                       output_names, target_names, debug_urls_);
}

Status DebugGraphDecorator::DecorateGraph(Graph* graph, Device* device) {
  DebugNodeInserter::DeparallelizeWhileLoops(graph, device);
  return DebugNodeInserter::InsertNodes(
      debug_options_.debug_tensor_watch_opts(), graph, device);
}

Status DebugGraphDecorator::PublishGraph(const Graph& graph,
                                         const string& device_name) {
  std::unordered_set<string> debug_urls;
  for (const DebugTensorWatch& watch :
       debug_options_.debug_tensor_watch_opts()) {
    for (const string& url : watch.debug_urls()) {
      debug_urls.insert(url);
    }
  }

  return DebugIO::PublishGraph(graph, device_name, debug_urls);
}

}  // namespace tensorflow
