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

#ifndef TENSORFLOW_CORE_DEBUG_DEBUGGER_STATE_IMPL_H_
#define TENSORFLOW_CORE_DEBUG_DEBUGGER_STATE_IMPL_H_

#include "tensorflow/core/common_runtime/debugger_state_interface.h"

#include <unordered_set>
#include <vector>

namespace tensorflow {

class DebuggerState : public DebuggerStateInterface {
 public:
  DebuggerState(const DebugOptions& debug_options);
  ~DebuggerState() override;

  // Publish metadata about the debugged Session::Run() call.
  //
  // See the doc string of DebuggerStateInterface::PublishDebugMetadata() for
  // details.
  absl::Status PublishDebugMetadata(
      const int64_t global_step, const int64_t session_run_count,
      const int64_t executor_step_count, const std::vector<string>& input_names,
      const std::vector<string>& output_names,
      const std::vector<string>& target_names) override;

 private:
  std::unordered_set<string> debug_urls_;
};

class DebugGraphDecorator : public DebugGraphDecoratorInterface {
 public:
  DebugGraphDecorator(const DebugOptions& debug_options)
      : debug_options_(debug_options) {}
  ~DebugGraphDecorator() override {}

  absl::Status DecorateGraph(Graph* graph, Device* device) override;
  absl::Status PublishGraph(const Graph& graph,
                            const string& device_name) override;

 private:
  DebugOptions debug_options_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DEBUG_DEBUGGER_STATE_IMPL_H_
