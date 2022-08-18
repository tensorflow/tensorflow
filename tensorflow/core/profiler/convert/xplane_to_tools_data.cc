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

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/hlo_to_tools_data.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/convert/op_stats_to_op_profile.h"
#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"
#include "tensorflow/core/profiler/convert/op_stats_to_pod_viewer.h"
#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"
#include "tensorflow/core/profiler/convert/tool_options.h"
#include "tensorflow/core/profiler/convert/xplane_to_hlo.h"
#include "tensorflow/core/profiler/convert/xplane_to_memory_profile.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_data_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"
#include "tensorflow/core/profiler/protobuf/pod_viewer.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_data_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

StatusOr<std::string> ConvertXSpaceToTraceEvents(
    const std::vector<XSpace>& xspaces) {
  if (xspaces.size() != 1) {
    return errors::InvalidArgument(
        "Trace events tool expects only 1 XSpace path but gets ",
        xspaces.size());
  }

  std::string content;
  ConvertXSpaceToTraceEventsString(xspaces[0], &content);
  return content;
}

StatusOr<std::string> ConvertMultiXSpacesToOverviewPage(
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
                                                          &combined_op_stats));
  OverviewPage overview_page_db;
  if (xspaces.size() == 1) {
    overview_page_db =
        ConvertOpStatsToOverviewPage(combined_op_stats, xspaces.at(0));
  } else {
    // TODO(profiler): xspace should tell whether this is sampling mode.
    overview_page_db = ConvertOpStatsToOverviewPage(combined_op_stats);
  }
  return overview_page_db.SerializeAsString();
}

StatusOr<std::string> ConvertMultiXSpacesToInputPipeline(
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
                                                          &combined_op_stats));
  return ConvertOpStatsToInputPipelineAnalysis(combined_op_stats)
      .SerializeAsString();
}

StatusOr<std::string> ConvertMultiXSpacesToTfStats(
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_kernel_stats_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
                                                          &combined_op_stats));
  return ConvertOpStatsToTfStats(combined_op_stats).SerializeAsString();
}

StatusOr<std::string> ConvertMultiXSpacesToKernelStats(
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
                                                          &combined_op_stats));
  return combined_op_stats.kernel_stats_db().SerializeAsString();
}

StatusOr<std::string> ConvertXSpaceToMemoryProfile(
    const std::vector<XSpace>& xspaces) {
  if (xspaces.size() != 1) {
    return errors::InvalidArgument(
        "Memory profile tool expects only 1 XSpace path but gets ",
        xspaces.size());
  }
  std::string json_output;
  TF_RETURN_IF_ERROR(
      ConvertXSpaceToMemoryProfileJson(xspaces[0], &json_output));
  return json_output;
}

StatusOr<std::string> ConvertMultiXSpacesToPodViewer(
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
                                                          &combined_op_stats));

  std::string json_output;
  protobuf::util::JsonPrintOptions opts;
  opts.always_print_primitive_fields = true;
  auto encode_status = protobuf::util::MessageToJsonString(
      ConvertOpStatsToPodViewer(combined_op_stats), &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return errors::Internal(
        "Could not convert pod viewer proto to json. Error: ",
        absl::string_view(error_message.data(), error_message.length()));
  }
  return json_output;
}

StatusOr<std::string> ConvertMultiXSpacesToTfDataBottleneckAnalysis(
    const std::vector<XSpace>& xspaces,
    const std::vector<std::string>& filenames) {
  CombinedTfDataStats combined_tf_data_stats;
  CombinedTfDataStatsBuilder builder(&combined_tf_data_stats);

  std::vector<XSpace> mutable_xspaces = xspaces;

  for (int idx = 0; idx < mutable_xspaces.size(); ++idx) {
    XPlane* host_plane =
        FindMutablePlaneWithName(&mutable_xspaces[idx], kHostThreadsPlaneName);
    if (host_plane == nullptr) {
      return errors::InvalidArgument(
          "Could not find host XPlane for tf data stats: ", filenames[idx]);
    }
    absl::string_view host_name = mutable_xspaces[idx].hostnames_size()
                                      ? mutable_xspaces[idx].hostnames(0)
                                      : filenames[idx];
    builder.Add(host_name, host_plane);
  }
  builder.Finalize();
  return combined_tf_data_stats.SerializeAsString();
}

StatusOr<std::string> ConvertMultiXSpacesToOpProfileViewer(
    const std::vector<XSpace>& xspaces) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(xspaces, options,
                                                          &combined_op_stats));

  tensorflow::profiler::op_profile::Profile profile;
  ConvertOpStatsToOpProfile(
      combined_op_stats,
      ParseHardwareType(combined_op_stats.run_environment().device_type()),
      profile);

  return profile.SerializeAsString();
}
}  // namespace

StatusOr<std::string> ConvertMultiXSpacesToToolData(
    const std::vector<XSpace>& xspaces,
    const std::vector<std::string>& filenames,
    const absl::string_view tool_name, const ToolOptions& options) {
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
    TF_RETURN_IF_ERROR(
        GetHloProtoFromMultiXSpaceAndSaveToFile(xspaces, filenames));
    return std::string();
  } else if (tool_name == "op_profile") {
    return ConvertMultiXSpacesToOpProfileViewer(xspaces);
  } else if (tool_name == "memory_viewer" || tool_name == "graph_viewer") {
    return ConvertHloProtoToToolData(filenames, tool_name, options);
  } else {
    return errors::InvalidArgument(
        "Can not find tool: ", tool_name,
        ". Please update to the latest version of Tensorflow.");
  }
}

}  // namespace profiler
}  // namespace tensorflow
