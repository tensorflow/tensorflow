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

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"
#include "tensorflow/core/profiler/convert/op_stats_to_pod_viewer.h"
#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_hlo.h"
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
    const std::vector<XSpace>& xspaces) {
  if (xspaces.size() != 1) {
    LOG(WARNING) << "Trace events tool expects only 1 XSpace path but gets "
                 << xspaces.size();
    return std::make_pair("", false);
  }

  std::string content;
  ConvertXSpaceToTraceEventsString(xspaces[0], &content);
  return std::make_pair(content, true);
}

std::pair<std::string, bool> ConvertMultiXSpacesToOverviewPage(
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  Status status = ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
                                                       &combined_op_stats);
  if (!status.ok()) {
    LOG(WARNING) << "Could not generate OpStats for overview page. Error: "
                 << status.error_message();
    return std::make_pair("", false);
  }
  OverviewPage overview_page_db;
  if (xspaces.size() == 1) {
    overview_page_db =
        ConvertOpStatsToOverviewPage(combined_op_stats, xspaces.at(0));
  } else {
    // TODO(profiler): xspace should tell whether this is sampling mode.
    overview_page_db = ConvertOpStatsToOverviewPage(combined_op_stats);
  }
  return std::make_pair(overview_page_db.SerializeAsString(), true);
}

std::pair<std::string, bool> ConvertMultiXSpacesToInputPipeline(
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  Status status = ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
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
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_kernel_stats_db = true;
  OpStats combined_op_stats;
  Status status = ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
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
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  OpStats combined_op_stats;
  Status status = ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
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
    const std::vector<XSpace>& xspaces) {
  if (xspaces.size() != 1) {
    LOG(WARNING) << "Memory profile tool expects only 1 XSpace path but gets "
                 << xspaces.size();
    return std::make_pair("", false);
  }
  std::string json_output;
  Status status;
  status = ConvertXSpaceToMemoryProfileJson(xspaces[0], &json_output);
  if (!status.ok()) {
    LOG(WARNING) << "Could not generate memory profile. Error: "
                 << status.error_message();
    return std::make_pair("", false);
  }
  return std::make_pair(json_output, true);
}

std::pair<std::string, bool> ConvertMultiXSpacesToPodViewer(
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  Status status = ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
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
    const std::vector<XSpace>& xspaces,
    const std::vector<std::string>& filenames) {
  CombinedTfDataStats combined_tf_data_stats;
  CombinedTfDataStatsBuilder builder(&combined_tf_data_stats);

  std::vector<XSpace> mutable_xspaces = xspaces;

  for (int idx = 0; idx < mutable_xspaces.size(); ++idx) {
    XPlane* host_plane =
        FindMutablePlaneWithName(&mutable_xspaces[idx], kHostThreadsPlaneName);
    if (host_plane == nullptr) {
      LOG(WARNING) << "Could not find host XPlane for tf data stats: ";
      return std::make_pair("", false);
    }
    absl::string_view host_name = mutable_xspaces[idx].hostnames_size()
                                      ? mutable_xspaces[idx].hostnames(0)
                                      : filenames[idx];
    builder.Add(host_name, host_plane);
  }
  builder.Finalize();
  return std::make_pair(combined_tf_data_stats.SerializeAsString(), true);
}

}  // namespace

std::pair<std::string, bool> ConvertMultiXSpacesToToolData(
    const std::vector<XSpace>& xspaces,
    const std::vector<std::string>& filenames,
    const absl::string_view tool_name) {
  if (tool_name == "trace_viewer") {
    return ConvertXSpaceToTraceEvents(xspaces);
  } else if (tool_name == "overview_page") {
    return ConvertMultiXSpacesToOverviewPage(xspaces);
  } else if (tool_name == "input_pipeline_analyzer") {
    return ConvertMultiXSpacesToInputPipeline(xspaces);
  } else if (tool_name == "tensorflow_stats") {
    return ConvertMultiXSpacesToTfStats(xspaces);
  } else if (tool_name == "kernel_stats") {
    return ConvertMultiXSpacesToKernelStats(xspaces);
  } else if (tool_name == "memory_profile") {
    return ConvertXSpaceToMemoryProfile(xspaces);
  } else if (tool_name == "pod_viewer") {
    return ConvertMultiXSpacesToPodViewer(xspaces);
  } else if (tool_name == "tf_data_bottleneck_analysis") {
    return ConvertMultiXSpacesToTfDataBottleneckAnalysis(xspaces, filenames);
  } else if (tool_name == "hlo_proto") {
    // <hlo_proto> is a special tool name to generate HLO proto files from
    // XSpace and store them in profile repository, this method does not return
    // actual tool data.
    auto status = GetHloProtoFromMultiXSpaceAndSaveToFile(xspaces, filenames);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to convert XSpace to HLO proto: "
                 << status.error_message();
      return std::make_pair("", false);
    }
    return std::make_pair("", true);
  } else {
    LOG(WARNING) << "Can not find tool: " << tool_name << ". Please update to "
                 << "the latest version of Tensorflow.";
    return std::make_pair("", false);
  }
}

}  // namespace profiler
}  // namespace tensorflow
