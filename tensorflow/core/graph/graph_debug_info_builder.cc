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

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stack_frame.h"
#include "tsl/platform/path.h"

namespace tensorflow {

// Ignore the frames containing this substring for common prefix calculation.
static const char* kFilenameToIgnorePrefix = "<embedded";

// Converts the given stack frame to a string.
std::string StackFrameToString(const StackFrame& frame,
                               int shared_prefix_length) {
  std::string out = absl::StrFormat(
      "File \"%s\", line %d, in %s",
      absl::StrContains(frame.file_name, kFilenameToIgnorePrefix)
          ? frame.file_name
          : frame.file_name.substr(shared_prefix_length),
      frame.line_number, frame.function_name);
  return out;
}

std::string ToStringHelper(absl::Span<const StackFrame> stack_frames,
                           int shared_prefix_length) {
  return absl::StrJoin(
      stack_frames, "\n", [&](std::string* out, const StackFrame& frame) {
        absl::StrAppend(out, StackFrameToString(frame, shared_prefix_length));
      });
}

FrozenStackTrace::FrozenStackTrace(absl::Span<StackFrame const> frames,
                                   absl::Span<StackFrame const> user_frames)
    : frames_(frames.begin(), frames.end()),
      user_frames_(user_frames.begin(), user_frames.end()) {
  if (user_frames.empty()) {
    user_frames_ = frames_;
  }
}

FrozenStackTrace::FrozenStackTrace(
    const GraphDebugInfo::StackTrace& stack_trace,
    const GraphDebugInfo& debug_info) {
  auto push_frame = [this,
                     &debug_info](const GraphDebugInfo::FileLineCol& frame) {
    int file_index = frame.file_index();
    std::string file_name =
        (file_index >= 0 && file_index < debug_info.files_size())
            ? debug_info.files(file_index)
            : "<UNKNOWN_FILE_NAME>";
    frames_.push_back(StackFrame(file_name, frame.line(), frame.func()));
  };

  if (!stack_trace.file_line_cols().empty()) {
    for (const GraphDebugInfo::FileLineCol& frame :
         stack_trace.file_line_cols()) {
      push_frame(frame);
    }
  } else {
    for (const uint64_t frame_id : stack_trace.frame_id()) {
      if (debug_info.frames_by_id().contains(frame_id)) {
        push_frame(debug_info.frames_by_id().at(frame_id));
      } else {
        LOG_FIRST_N(ERROR, 5) << "No matching frame for id:" << frame_id;
      }
    }
  }
}

absl::Span<StackFrame const> FrozenStackTrace::ToFrames() const {
  return frames_;
}

StackFrame FrozenStackTrace::LastUserFrame() const { return frames_.back(); }

std::vector<StackFrame> FrozenStackTrace::GetUserFrames(int limit) const {
  std::vector<StackFrame> result;
  if (limit < 0 || limit > user_frames_.size()) {
    limit = user_frames_.size();
  }
  result.reserve(limit);
  for (int i = 0; i < limit; ++i) {
    result.push_back(user_frames_[i]);
  }
  return result;
}

std::string FrozenStackTrace::ToString(const TracePrintingOptions& opts) const {
  int shared_prefix_length = 0;
  if (opts.filter_common_prefix) {
    std::vector<std::string> prefix_file_names;
    for (const StackFrame& frame : frames_) {
      if (!absl::StrContains(frame.file_name, kFilenameToIgnorePrefix)) {
        prefix_file_names.push_back(frame.file_name);
      }
    }
    shared_prefix_length = tsl::io::CommonPathPrefix(prefix_file_names).size();
  }

  if (!opts.drop_internal_frames) {
    return ToStringHelper(frames_, shared_prefix_length);
  }

  std::vector<StackFrame> non_internal_frames;
  for (const StackFrame& frame : frames_) {
    if (!IsInternalFrameForFilename(frame.file_name)) {
      non_internal_frames.push_back(frame);
    }
  }
  return ToStringHelper(non_internal_frames, shared_prefix_length);
}

GraphDebugInfoBuilder::GraphDebugInfoBuilder()
    : debug_info_(std::make_unique<GraphDebugInfo>()) {}

void GraphDebugInfoBuilder::AccumulateStackTracesMap(
    const StackTracesMap& stack_traces_map, absl::string_view key_suffix,
    const GraphDebugInfoBuilder::Options& options) {
  trace_to_index_.reserve(trace_to_index_.size() + stack_traces_map.size());
  for (const auto& [node_name, stack_trace] : stack_traces_map) {
    if (stack_trace == nullptr) continue;
    std::string trace_key = absl::StrCat(node_name, key_suffix);
    AccumulateStackTrace(stack_trace, trace_key, options);
  }
}

void GraphDebugInfoBuilder::AccumulateStackTrace(
    std::shared_ptr<AbstractStackTrace> trace, absl::string_view traces_key,
    const GraphDebugInfoBuilder::Options& options) {
  int trace_index = 0;
  StackTracePointer p{trace};
  auto found = trace_to_index_.find(p);
  if (found != trace_to_index_.end()) {
    trace_index = found->second;
  } else {
    trace_index = debug_info_->traces_by_id().size();
    trace_to_index_[p] = trace_index;
    GraphDebugInfo::StackTrace& stack_trace_proto =
        (*debug_info_->mutable_traces_by_id())[trace_index];
    if (options.user_frames) {
      frame_to_index_.reserve(
          frame_to_index_.size() +
          trace->GetUserFrames(options.user_frames_limit).size());
      for (const auto& stack_frame :
           trace->GetUserFrames(options.user_frames_limit)) {
        AppendToStackTraceProto(stack_frame, stack_trace_proto);
      }
    } else {
      frame_to_index_.reserve(frame_to_index_.size() +
                              trace->ToFrames().size());
      for (const auto& stack_frame : trace->ToFrames()) {
        AppendToStackTraceProto(stack_frame, stack_trace_proto);
      }
    }
  }
  (*debug_info_->mutable_name_to_trace_id())[traces_key] = trace_index;
}

void GraphDebugInfoBuilder::AppendToStackTraceProto(
    const StackFrame& stack_frame,
    GraphDebugInfo::StackTrace& stack_trace_proto) {
  int frame_index = 0;
  auto found = frame_to_index_.find(stack_frame);
  if (found != frame_to_index_.end()) {
    frame_index = found->second;
  } else {
    frame_index = debug_info_->frames_by_id().size();
    frame_to_index_[stack_frame] = frame_index;
    GraphDebugInfo::FileLineCol& frame =
        (*debug_info_->mutable_frames_by_id())[frame_index];
    auto file_index = file_name_to_index_.find(stack_frame.file_name);
    if (file_index != file_name_to_index_.end()) {
      frame.set_file_index(file_index->second);
    } else {
      frame.set_file_index(new_name_index_);
      file_name_to_index_[stack_frame.file_name] = new_name_index_;
      *debug_info_->add_files() = stack_frame.file_name;
      new_name_index_++;
    }
    frame.set_line(stack_frame.line_number);
    frame.set_func(stack_frame.function_name);
  }
  stack_trace_proto.add_frame_id(frame_index);
}

void GraphDebugInfoBuilder::AppendGraphDebugInfo(
    absl::string_view prefix, const GraphDebugInfo& new_info) {
  for (const auto& pair : new_info.name_to_trace_id()) {
    auto trace = new_info.traces_by_id().at(pair.second);
    auto frozen = std::make_shared<FrozenStackTrace>(trace, new_info);
    std::string key =
        prefix.empty() ? pair.first : absl::StrCat(pair.first, "@", prefix);
    AccumulateStackTrace(frozen, key, GraphDebugInfoBuilder::Options{});
  }
}

GraphDebugInfo GraphDebugInfoBuilder::Build() const { return *debug_info_; }

absl::Status GraphDebugInfoBuilder::AppendGraphDebugInfoStr(
    absl::string_view prefix, absl::string_view new_info_str) {
  GraphDebugInfo debug_info;
  if (!debug_info.ParseFromArray(new_info_str.data(), new_info_str.size())) {
    return absl::InvalidArgumentError("Failed to parse GraphDebugInfo proto.");
  }
  AppendGraphDebugInfo(prefix, debug_info);
  return absl::OkStatus();
}

std::string GraphDebugInfoBuilder::ToGraphDebugInfoStr() const {
  return Build().SerializeAsString();
}

StackTracesMap LoadTracesFromDebugInfo(const GraphDebugInfo& debug_info) {
  StackTracesMap traces;
  absl::flat_hash_map<uint64_t, std::shared_ptr<AbstractStackTrace>>
      traces_by_id;
  traces_by_id.reserve(debug_info.traces_by_id_size());
  for (const auto& [id, frames] : debug_info.traces_by_id()) {
    traces_by_id[id] = std::make_shared<FrozenStackTrace>(frames, debug_info);
  }

  traces.reserve(debug_info.name_to_trace_id_size() + debug_info.traces_size());
  for (const auto& [name, trace_id] : debug_info.name_to_trace_id()) {
    if (!traces_by_id.contains(trace_id)) {
      LOG_FIRST_N(ERROR, 5) << "No matching trace for id:" << trace_id;
      continue;
    }
    traces[name] = traces_by_id[trace_id];
  }

  for (const auto& [name, frames] : debug_info.traces()) {
    traces[name] = std::make_shared<FrozenStackTrace>(frames, debug_info);
  }

  return traces;
}

absl::StatusOr<StackTracesMap> LoadTracesFromDebugInfoStr(
    absl::string_view debug_info_str) {
  GraphDebugInfo debug_info;
  if (!debug_info.ParseFromArray(debug_info_str.data(),
                                 debug_info_str.size())) {
    return absl::InvalidArgumentError("Failed to parse GraphDebugInfo proto.");
  }
  return LoadTracesFromDebugInfo(debug_info);
}

GraphDebugInfo StackTracesMapToGraphDebugInfo(const StackTracesMap& map,
                                              bool user_frames) {
  GraphDebugInfoBuilder builder;
  GraphDebugInfoBuilder::Options options;
  options.user_frames = user_frames;
  options.user_frames_limit = -1;
  builder.AccumulateStackTracesMap(map, "", options);
  return builder.Build();
}

}  // namespace tensorflow
