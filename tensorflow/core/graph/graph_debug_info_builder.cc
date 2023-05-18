/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/graph_debug_info_builder.h"

#include <string>
#include <utility>

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/platform/stack_frame.h"

namespace tensorflow {

void GraphDebugInfoBuilder::AccumulateStackTracesMap(
    const StackTracesMap& stack_traces_map, absl::string_view key_suffix,
    const GraphDebugInfoBuilder::Options& options) {
  for (const auto& [node_name, stack_trace] : stack_traces_map) {
    std::string trace_key = absl::StrCat(node_name, key_suffix);
    AccumulateStackTrace(*stack_trace, trace_key, options);
  }
}

void GraphDebugInfoBuilder::AccumulateStackTrace(
    const AbstractStackTrace& abstract_stack_trace,
    absl::string_view traces_key,
    const GraphDebugInfoBuilder::Options& options) {
  GraphDebugInfo::StackTrace stack_trace_proto;
  if (options.user_frames) {
    for (const auto& stack_frame :
         abstract_stack_trace.GetUserFrames(options.user_frames_limit)) {
      AppendToStackTraceProto(stack_frame, stack_trace_proto);
    }
  } else {
    for (const auto& stack_frame : abstract_stack_trace.ToFrames()) {
      AppendToStackTraceProto(stack_frame, stack_trace_proto);
    }
  }
  (*debug_info_.mutable_traces())[traces_key] = std::move(stack_trace_proto);
}

void GraphDebugInfoBuilder::AppendToStackTraceProto(
    const StackFrame& stack_frame,
    GraphDebugInfo::StackTrace& stack_trace_proto) {
  auto& file_line_col = *stack_trace_proto.add_file_line_cols();
  if (file_name_to_index_.contains(stack_frame.file_name)) {
    file_line_col.set_file_index(file_name_to_index_[stack_frame.file_name]);
  } else {
    file_line_col.set_file_index(new_name_index_);
    file_name_to_index_[stack_frame.file_name] = new_name_index_;
    *debug_info_.add_files() = stack_frame.file_name;
    new_name_index_++;
  }
  file_line_col.set_line(stack_frame.line_number);
  file_line_col.set_func(stack_frame.function_name);
}

GraphDebugInfo GraphDebugInfoBuilder::Build() const { return debug_info_; }

}  // namespace tensorflow
