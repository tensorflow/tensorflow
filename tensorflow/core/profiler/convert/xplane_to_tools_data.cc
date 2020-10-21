/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/xplane_to_tools_data.h"

#include <utility>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"
#include "tensorflow/core/profiler/convert/op_stats_to_pod_viewer.h"
#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_memory_profile.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_data_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"
#include "tensorflow/core/profiler/protobuf/pod_viewer.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_data_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

std::pair<std::string, bool> ConvertXSpaceToTraceEvents(
    const std::vector<std::string>& xspace_paths) {
  if (xspace_paths.size() != 1) {
    LOG(WARNING) << "Trace events tool expects only 1 XSpace path but gets "
                 << xspace_paths.size();
    return std::make_pair("", false);
  }

  XSpace xspace;
  Status status = ReadBinaryProto(Env::Default(), xspace_paths[0], &xspace);
  if (!status.ok()) {
    LOG(WARNING) << "Could not read XSpace for trace events: "
                 << xspace_paths[0];
    return std::make_pair("", false);
  }
  std::string content;
  ConvertXSpaceToTraceEventsString(xspace, &content);
  return std::make_pair(content, true);
}

std::pair<std::string, bool> ConvertMultiXSpacesToOverviewPage(
    const std::vector<std::string>& xspace_paths) {
  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  Status status = ConvertMultiXSpacesToCombinedOpStats(xspace_paths, options,
                                                       &combined_op_stats);
  if (!status.ok()) {
    LOG(WARNING) << "Could not generate OpStats for overview page. Error: "
                 << status.error_message();
    return std::make_pair("", false);
  }
  // TODO(profiler): xspace should tell whether this is sampling mode.
  return std::make_pair(
      ConvertOpStatsToOverviewPage(combined_op_stats).SerializeAsString(),
      true);
}

std::pair<std::string, bool> ConvertMultiXSpacesToInputPipeline(
    const std::vector<std::string>& xspace_paths) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  Status status = ConvertMultiXSpacesToCombinedOpStats(xspace_paths, options,
                                                       &combined_op_stats);
  if (!status.ok()) {
    LOG(WARNING) << "Could not generate OpStats for input pipeline. Error: "
                 << status.error_message();
    return std::make_pair("", false);
  }
  return std::make_pair(ConvertOpStatsToInputPipelineAnalysis(combined_op_stats)
                            .SerializeAsString(),
                        true);
}

std::pair<std::string, bool> ConvertMultiXSpacesToTfStats(
    const std::vector<std::string>& xspace_paths) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_kernel_stats_db = true;
  OpStats combined_op_stats;
  Status status = ConvertMultiXSpacesToCombinedOpStats(xspace_paths, options,
                                                       &combined_op_stats);
  if (!status.ok()) {
    LOG(WARNING) << "Could not generate OpStats for tensorflow stats. Error: "
                 << status.error_message();
    return std::make_pair("", false);
  }
  return std::make_pair(
      ConvertOpStatsToTfStats(combined_op_stats).SerializeAsString(), true);
}

std::pair<std::string, bool> ConvertMultiXSpacesToKernelStats(
    const std::vector<std::string>& xspace_paths) {
  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  OpStats combined_op_stats;
  Status status = ConvertMultiXSpacesToCombinedOpStats(xspace_paths, options,
                                                       &combined_op_stats);
  if (!status.ok()) {
    LOG(WARNING) << "Could not generate OpStats for kernel stats. Error: "
                 << status.error_message();
    return std::make_pair("", false);
  }
  return std::make_pair(combined_op_stats.kernel_stats_db().SerializeAsString(),
                        true);
}

std::pair<std::string, bool> ConvertXSpaceToMemoryProfile(
    const std::vector<std::string>& xspace_paths) {
  if (xspace_paths.size() != 1) {
    LOG(WARNING) << "Memory profile tool expects only 1 XSpace path but gets "
                 << xspace_paths.size();
    return std::make_pair("", false);
  }
  XSpace xspace;
  Status status = ReadBinaryProto(Env::Default(), xspace_paths[0], &xspace);
  if (!status.ok()) {
    LOG(WARNING) << "Could not read XSpace for memory profile: "
                 << xspace_paths[0];
    return std::make_pair("", false);
  }
  std::string json_output;
  status = ConvertXSpaceToMemoryProfileJson(xspace, &json_output);
  if (!status.ok()) {
    LOG(WARNING) << "Could not generate memory profile. Error: "
                 << status.error_message();
    return std::make_pair("", false);
  }
  return std::make_pair(json_output, true);
}

std::pair<std::string, bool> ConvertMultiXSpacesToPodViewer(
    const std::vector<std::string>& xspace_paths) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  Status status = ConvertMultiXSpacesToCombinedOpStats(xspace_paths, options,
                                                       &combined_op_stats);
  if (!status.ok()) {
    LOG(WARNING) << "Could not generate OpStats for pod_viewer. Error: "
                 << status.error_message();
    return std::make_pair("", false);
  }

  std::string json_output;
  protobuf::util::JsonPrintOptions opts;
  opts.always_print_primitive_fields = true;
  auto encode_status = protobuf::util::MessageToJsonString(
      ConvertOpStatsToPodViewer(combined_op_stats), &json_output, opts);
  if (!encode_status.ok()) {
    LOG(WARNING) << "Could not convert pod viewer proto to json. Error: "
                 << encode_status.message();
    return std::make_pair("", false);
  }
  return std::make_pair(json_output, true);
}

std::pair<std::string, bool> ConvertMultiXSpacesToTfDataBottleneckAnalysis(
    const std::vector<std::string>& xspace_paths) {
  CombinedTfDataStats combined_tf_data_stats;
  CombinedTfDataStatsBuilder builder(&combined_tf_data_stats);
  for (const std::string& xspace_path : xspace_paths) {
    XSpace xspace;
    Status status = ReadBinaryProto(Env::Default(), xspace_path, &xspace);
    if (!status.ok()) {
      LOG(WARNING) << "Could not read XSpace for tf data stats: "
                   << xspace_path;
      return std::make_pair("", false);
    }
    XPlane* host_plane =
        FindMutablePlaneWithName(&xspace, kHostThreadsPlaneName);
    if (host_plane == nullptr) {
      LOG(WARNING) << "Could not find host XPlane for tf data stats: "
                   << xspace_path;
      return std::make_pair("", false);
    }
    absl::string_view host_name =
        xspace.hostnames_size() ? xspace.hostnames(0) : xspace_path;
    builder.Add(host_name, host_plane);
  }
  builder.Finalize();
  return std::make_pair(combined_tf_data_stats.SerializeAsString(), true);
}

}  // namespace

std::pair<std::string, bool> ConvertMultiXSpacesToToolData(
    const std::vector<std::string>& xspace_paths,
    const absl::string_view tool_name) {
  if (tool_name == "trace_viewer") {
    return ConvertXSpaceToTraceEvents(xspace_paths);
  } else if (tool_name == "overview_page") {
    return ConvertMultiXSpacesToOverviewPage(xspace_paths);
  } else if (tool_name == "input_pipeline_analyzer") {
    return ConvertMultiXSpacesToInputPipeline(xspace_paths);
  } else if (tool_name == "tensorflow_stats") {
    return ConvertMultiXSpacesToTfStats(xspace_paths);
  } else if (tool_name == "kernel_stats") {
    return ConvertMultiXSpacesToKernelStats(xspace_paths);
  } else if (tool_name == "memory_profile") {
    return ConvertXSpaceToMemoryProfile(xspace_paths);
  } else if (tool_name == "pod_viewer") {
    return ConvertMultiXSpacesToPodViewer(xspace_paths);
  } else if (tool_name == "tf_data_bottleneck_analysis") {
    return ConvertMultiXSpacesToTfDataBottleneckAnalysis(xspace_paths);
  } else {
    LOG(WARNING) << "Can not find tool: " << tool_name << ". Please update to "
                 << "the latest version of Tensorflow.";
    return std::make_pair("", false);
  }
}

}  // namespace profiler
}  // namespace tensorflow
