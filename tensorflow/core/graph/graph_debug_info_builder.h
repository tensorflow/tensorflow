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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/platform/stack_frame.h"
#include "tsl/platform/macros.h"

namespace tensorflow {

// Language agnostic stack traces.
class AbstractStackTrace {
 public:
  struct TracePrintingOptions {
    // Show inline the contents of each stack line.
    bool show_line_contents = false;

    // Drop the common largest prefix of all filenames in stack frames.
    bool filter_common_prefix = false;

    // Do not show internal frames.
    bool drop_internal_frames = false;
  };

  virtual ~AbstractStackTrace() = default;

  // The returned span is alive as long as the AbstractStackTrace is alive.
  virtual absl::Span<StackFrame const> ToFrames() const = 0;

  // Returns the stack frames without caching any generated data.
  virtual std::vector<StackFrame> ToUncachedFrames() const = 0;

  // Returns the last stack frame from user code, attempting to ignore the
  // framework code. Returns an empty frame if no such stack frame was found.
  virtual StackFrame LastUserFrame() const = 0;

  // Returns stack trace from user code (instead of op creation ones returned in
  // ToFrames).
  virtual std::vector<StackFrame> GetUserFrames(int limit) const = 0;

  virtual std::string ToString(const TracePrintingOptions& opts) const = 0;
};

// A frozen sequence of StackFrames; an adapter for a span of StackFrames that
// conforms to the AbstractStackTrace contract.
class FrozenStackTrace : public AbstractStackTrace {
 public:
  // Constructs a FrozenStackTrace from a span of StackFrames by making a copy
  // of each stack frame.
  explicit FrozenStackTrace(absl::Span<StackFrame const> frames,
                            absl::Span<StackFrame const> user_frames = {});

  explicit FrozenStackTrace(std::vector<StackFrame>&& frames)
      : frames_(std::move(frames)), user_frames_({}) {}

  FrozenStackTrace(FrozenStackTrace&&) = default;

  // Constructs a FrozenStackTrace from serialized proto data.
  FrozenStackTrace(const GraphDebugInfo::StackTrace& stack_trace,
                   const GraphDebugInfo& debug_info);

  ~FrozenStackTrace() override = default;

  absl::Span<StackFrame const> ToFrames() const override;

  std::vector<StackFrame> ToUncachedFrames() const override;

  StackFrame LastUserFrame() const override;

  std::vector<StackFrame> GetUserFrames(int limit) const override;

  std::string ToString(const TracePrintingOptions& opts) const override;

 private:
  std::vector<StackFrame> frames_;
  std::vector<StackFrame> user_frames_;
};

// Holder type to use `AbstractStackTrace` as a key.
struct StackTracePointer {
  std::shared_ptr<AbstractStackTrace> trace;

  template <class H>
  friend H AbslHashValue(H h, const StackTracePointer& p) {
    for (const auto& frame : p.trace->ToFrames()) {
      h = H::combine(std::move(h), frame);
    }
    return h;
  }

  bool operator==(const StackTracePointer& other) const {
    absl::Span<StackFrame const> other_frames = other.trace->ToFrames();
    absl::Span<StackFrame const> frames = trace->ToFrames();
    return frames == other_frames;
  }
};

using StackTracesMap =
    absl::flat_hash_map<std::string,
                        std::shared_ptr<tensorflow::AbstractStackTrace>>;

// Load all stack traces from `debug_info`.
StackTracesMap LoadTracesFromDebugInfo(const GraphDebugInfo& debug_info);
absl::StatusOr<StackTracesMap> LoadTracesFromDebugInfoStr(
    absl::string_view debug_info_str);

// Generates a GraphDebugInfo proto from a StackTracesMap object. Returns user
// frames by default. If `user_frames` is false, returns all frames.
GraphDebugInfo StackTracesMapToGraphDebugInfo(const StackTracesMap& map,
                                              bool user_frames = true);

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

  GraphDebugInfoBuilder();
  virtual ~GraphDebugInfoBuilder() = default;

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
  void AccumulateStackTrace(std::shared_ptr<AbstractStackTrace> trace,
                            absl::string_view traces_key,
                            const GraphDebugInfoBuilder::Options& options =
                                GraphDebugInfoBuilder::Options());

  void AppendGraphDebugInfo(absl::string_view prefix,
                            const GraphDebugInfo& new_info);

  // These string methods are used in the Python bindings  to avoid symbol
  // resolution errors with pybind on Windows.
  absl::Status AppendGraphDebugInfoStr(absl::string_view prefix,
                                       absl::string_view new_info_str);

  std::string ToGraphDebugInfoStr() const;

  // Returns the GraphDebugInfo proto.
  GraphDebugInfo Build() const;

 private:
  void AppendToStackTraceProto(const StackFrame& stack_frame,
                               GraphDebugInfo::StackTrace& stack_trace_proto);

  std::unique_ptr<GraphDebugInfo> debug_info_;
  absl::flat_hash_map<std::string, int> file_name_to_index_;

  absl::flat_hash_map<StackTracePointer, int> trace_to_index_;
  absl::flat_hash_map<StackFrame, int> frame_to_index_;
  int new_name_index_ = 0;

  GraphDebugInfoBuilder(const GraphDebugInfoBuilder&) = delete;
  void operator=(const GraphDebugInfoBuilder&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_GRAPH_DEBUG_INFO_BUILDER_H_
