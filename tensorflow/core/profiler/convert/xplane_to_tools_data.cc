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

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/hlo_to_tools_data.h"
#include "tensorflow/core/profiler/convert/multi_xplanes_to_op_stats.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/convert/op_stats_to_op_profile.h"
#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"
#include "tensorflow/core/profiler/convert/op_stats_to_pod_viewer.h"
#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"
#include "tensorflow/core/profiler/convert/preprocess_single_host_xplane.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/convert/tool_options.h"
#include "tensorflow/core/profiler/convert/xplane_to_memory_profile.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_data_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"
#include "tensorflow/core/profiler/convert/xplane_to_tool_names.h"
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
#include "tensorflow/tsl/profiler/convert/xplane_to_trace_events.h"

namespace tensorflow {
namespace profiler {

namespace {

StatusOr<std::string> ConvertXSpaceToTraceEvents(
    const SessionSnapshot& session_snapshot) {
  if (session_snapshot.XSpaceSize() != 1) {
    return errors::InvalidArgument(
        "Trace events tool expects only 1 XSpace path but gets ",
        session_snapshot.XSpaceSize());
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                      session_snapshot.GetXSpace(0));
  PreprocessSingleHostXSpace(xspace.get(), /*step_grouping=*/true,
                             /*derived_timeline=*/true);
  std::string content;
  tsl::profiler::ConvertXSpaceToTraceEventsString(*xspace, &content);
  return content;
}

StatusOr<std::string> ConvertMultiXSpacesToOverviewPage(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));
  // TODO(profiler): xspace should tell whether this is sampling mode.
  return ConvertOpStatsToOverviewPage(combined_op_stats).SerializeAsString();
}

StatusOr<std::string> ConvertMultiXSpacesToInputPipeline(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));
  return ConvertOpStatsToInputPipelineAnalysis(combined_op_stats)
      .SerializeAsString();
}

StatusOr<std::string> ConvertMultiXSpacesToTfStats(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_kernel_stats_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));
  return ConvertOpStatsToTfStats(combined_op_stats).SerializeAsString();
}

StatusOr<std::string> ConvertMultiXSpacesToKernelStats(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));
  return combined_op_stats.kernel_stats_db().SerializeAsString();
}

StatusOr<std::string> ConvertXSpaceToMemoryProfile(
    const SessionSnapshot& session_snapshot) {
  if (session_snapshot.XSpaceSize() != 1) {
    return errors::InvalidArgument(
        "Memory profile tool expects only 1 XSpace path but gets ",
        session_snapshot.XSpaceSize());
  }

  std::string json_output;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                      session_snapshot.GetXSpace(0));
  PreprocessSingleHostXSpace(xspace.get(), /*step_grouping=*/true,
                             /*derived_timeline=*/false);
  TF_RETURN_IF_ERROR(ConvertXSpaceToMemoryProfileJson(*xspace, &json_output));
  return json_output;
}

StatusOr<std::string> ConvertMultiXSpacesToPodViewer(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));

  std::string json_output;
  protobuf::util::JsonPrintOptions opts;
  opts.always_print_primitive_fields = true;
  auto encode_status = protobuf::util::MessageToJsonString(
      ConvertOpStatsToPodViewer(combined_op_stats), &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return errors::Internal(
        "Could not convert pod viewer to json. Error: ",
        absl::string_view(error_message.data(), error_message.length()));
  }
  return json_output;
}

StatusOr<std::string> ConvertMultiXSpacesToTfDataBottleneckAnalysis(
    const SessionSnapshot& session_snapshot) {
  CombinedTfDataStats combined_tf_data_stats;
  CombinedTfDataStatsBuilder builder(&combined_tf_data_stats);

  for (int idx = 0; idx < session_snapshot.XSpaceSize(); ++idx) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                        session_snapshot.GetXSpace(idx));

    PreprocessSingleHostXSpace(xspace.get(), /*step_grouping=*/true,
                               /*derived_timeline=*/false);
    XPlane* host_plane =
        FindMutablePlaneWithName(xspace.get(), kHostThreadsPlaneName);
    std::string host_name_from_file = session_snapshot.GetHostname(idx);
    if (host_plane == nullptr) {
      return errors::InvalidArgument(
          "Could not find host XPlane for tf data stats: ",
          host_name_from_file);
    }
    absl::string_view host_name =
        xspace->hostnames_size() ? xspace->hostnames(0) : host_name_from_file;
    builder.Add(host_name, host_plane);
  }
  builder.Finalize();
  return combined_tf_data_stats.SerializeAsString();
}

StatusOr<std::string> ConvertMultiXSpacesToOpProfileViewer(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));

  tensorflow::profiler::op_profile::Profile profile;
  ConvertOpStatsToOpProfile(
      combined_op_stats,
      ParseHardwareType(combined_op_stats.run_environment().device_type()),
      profile);
  std::string json_output;
  protobuf::util::JsonPrintOptions opts;
  opts.always_print_primitive_fields = true;

  auto encode_status =
      protobuf::util::MessageToJsonString(profile, &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return errors::Internal(
        "Could not convert op profile proto to json. Error: ",
        absl::string_view(error_message.data(), error_message.length()));
  }
  return json_output;
}

StatusOr<std::string> PreprocessXSpace(
    const SessionSnapshot& session_snapshot) {
  if (session_snapshot.XSpaceSize() != 1) {
    return errors::InvalidArgument(
        "PreprocessXSpace tool expects only 1 XSpace path but gets ",
        session_snapshot.XSpaceSize());
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                      session_snapshot.GetXSpace(0));
  PreprocessSingleHostXSpace(xspace.get(), /*step_grouping=*/true,
                             /*derived_timeline=*/true);
  return xspace->SerializeAsString();
}

}  // namespace

StatusOr<std::string> ConvertMultiXSpacesToToolData(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
  LOG(INFO) << "serving tool: " << tool_name
            << " with options: " << DebugString(options);
  if (tool_name == "trace_viewer") {
    return ConvertXSpaceToTraceEvents(session_snapshot);
  } else if (tool_name == "overview_page") {
    return ConvertMultiXSpacesToOverviewPage(session_snapshot);
  } else if (tool_name == "input_pipeline_analyzer") {
    return ConvertMultiXSpacesToInputPipeline(session_snapshot);
  } else if (tool_name == "tensorflow_stats") {
    return ConvertMultiXSpacesToTfStats(session_snapshot);
  } else if (tool_name == "kernel_stats") {
    return ConvertMultiXSpacesToKernelStats(session_snapshot);
  } else if (tool_name == "memory_profile") {
    return ConvertXSpaceToMemoryProfile(session_snapshot);
  } else if (tool_name == "pod_viewer") {
    return ConvertMultiXSpacesToPodViewer(session_snapshot);
  } else if (tool_name == "tf_data_bottleneck_analysis") {
    return ConvertMultiXSpacesToTfDataBottleneckAnalysis(session_snapshot);
  } else if (tool_name == "op_profile") {
    return ConvertMultiXSpacesToOpProfileViewer(session_snapshot);
  } else if (tool_name == "memory_viewer" || tool_name == "graph_viewer") {
    return ConvertHloProtoToToolData(session_snapshot, tool_name, options);
  } else if (tool_name == "tool_names") {
    return GetAvailableToolNames(session_snapshot);
  } else if (tool_name == "_xplane.pb") {  // internal test only.
    return PreprocessXSpace(session_snapshot);
  } else {
    return errors::InvalidArgument(
        "Can not find tool: ", tool_name,
        ". Please update to the latest version of Tensorflow.");
  }
}

}  // namespace profiler
}  // namespace tensorflow
