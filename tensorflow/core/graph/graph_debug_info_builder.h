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

#ifndef TENSORFLOW_CORE_GRAPH_GRAPH_DEBUG_INFO_BUILDER_H_
#define TENSORFLOW_CORE_GRAPH_GRAPH_DEBUG_INFO_BUILDER_H_

#include <string>

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/platform/stack_frame.h"
#include "tensorflow/tsl/platform/macros.h"

namespace tensorflow {

// Builder for GraphDebugInfo protos from either an existing map of string keys
// to stack traces, or individual stack traces, or both. All stack traces in a
// GraphDebugInfo are stored with a string key in the `traces` field. In the
// case of an existing map, its keys are used, appended with a key suffix,
// which may be empty. If it is not empty, it is conventionally of the form
// "@function_name", although this class doesn't care. In the case of an
// individual stack trace, a key for `traces` must be provided.
//
// This builder will create a list of the unique file names across all stack
// traces and store it in the `files` field. When storing stack traces into the
// proto, file names are replaced by their index into `files`.
//
// Typical usage is to call one or both of the accumulate methods one or more
// times and then to call the Build().

class GraphDebugInfoBuilder {
 public:
  struct Options {
    // Call the AbstractTraceMap GetUserFrames method rather than ToFrames
    bool user_frames;
    // Value of `limit` to pass to GetUserFrames if `user_frames` is true,
    // otherwise ignored
    int user_frames_limit;
  };

  explicit GraphDebugInfoBuilder() = default;

  // Adds a map of stack traces to the GraphDebugInfo proto. For each key (node
  // id) and stack traces entry in `stack_traces_map`, combine the key with
  // `key_suffix` to form a new key and use that to add the stack traces to the
  // `traces` field of the proto. If not empty, the suffix is typically of the
  // form "@function_name", although this function doesn't care.
  void AccumulateStackTracesMap(const StackTracesMap& stack_traces_map,
                                absl::string_view key_suffix = "",
                                const GraphDebugInfoBuilder::Options& options =
                                    GraphDebugInfoBuilder::Options());

  // Adds one stack trace to the GraphDebugInfo proto, using `traces_key` as the
  // key for the `traces` field of the proto.
  void AccumulateStackTrace(const AbstractStackTrace& abstract_stack_trace,
                            absl::string_view traces_key,
                            const GraphDebugInfoBuilder::Options& options =
                                GraphDebugInfoBuilder::Options());

  // Returns the GraphDebugInfo proto.
  GraphDebugInfo Build() const;

 private:
  void AppendToStackTraceProto(const StackFrame& stack_frame,
                               GraphDebugInfo::StackTrace& stack_trace_proto);

  GraphDebugInfo debug_info_;
  absl::flat_hash_map<std::string, int> file_name_to_index_;
  int new_name_index_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphDebugInfoBuilder);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_GRAPH_DEBUG_INFO_BUILDER_H_
