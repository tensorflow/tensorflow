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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/profiler/convert/compute_inference_latency.h"
#include "tensorflow/core/profiler/convert/hlo_to_tools_data.h"
#include "tensorflow/core/profiler/convert/multi_xplanes_to_op_stats.h"
#include "tensorflow/core/profiler/convert/multi_xspace_to_inference_stats.h"
#include "tensorflow/core/profiler/convert/op_stats_to_hlo_stats.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/convert/op_stats_to_op_profile.h"
#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"
#include "tensorflow/core/profiler/convert/op_stats_to_pod_viewer.h"
#include "tensorflow/core/profiler/convert/op_stats_to_roofline_model.h"
#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"
#include "tensorflow/core/profiler/convert/preprocess_single_host_xplane.h"
#include "tensorflow/core/profiler/convert/process_megascale_dcn.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/convert/tool_options.h"
#include "tensorflow/core/profiler/convert/xplane_to_dcn_collective_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_memory_profile.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_data_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_tool_names.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_container.h"
#include "tensorflow/core/profiler/protobuf/dcn_slack_analysis.pb.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"
#include "tensorflow/core/profiler/protobuf/roofline_model.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_data_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"  // from @org_xprof
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"  // from @org_xprof
#include "xprof/utils/hardware_type_utils.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {

namespace {

struct TraceViewOption {
  uint64_t resolution = 0;
  double start_time_ms = 0.0;
  double end_time_ms = 0.0;
};

absl::StatusOr<TraceViewOption> GetTraceViewOption(const ToolOptions& options) {
  TraceViewOption trace_options;
  auto start_time_ms_opt =
      GetParamWithDefault<std::string>(options, "start_time_ms", "0.0");
  auto end_time_ms_opt =
      GetParamWithDefault<std::string>(options, "end_time_ms", "0.0");
  auto resolution_opt =
      GetParamWithDefault<std::string>(options, "resolution", "0");

  if (!absl::SimpleAtoi(resolution_opt, &trace_options.resolution) ||
      !absl::SimpleAtod(start_time_ms_opt, &trace_options.start_time_ms) ||
      !absl::SimpleAtod(end_time_ms_opt, &trace_options.end_time_ms)) {
    return errors::InvalidArgument("wrong arguments");
  }
  return trace_options;
}

absl::StatusOr<std::string> ConvertXSpaceToTraceEvents(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
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
  if (tool_name == "trace_viewer") {
    tsl::profiler::ConvertXSpaceToTraceEventsString(*xspace, &content);
    return content;
  } else {  // streaming trace viewer.
    std::string host_name = session_snapshot.GetHostname(0);
    auto sstable_path = session_snapshot.GetFilePath(tool_name, host_name);
    if (!sstable_path) {
      return errors::Unimplemented(
          "streaming trace viewer hasn't been supported in Cloud AI");
    }
    if (!Env::Default()->FileExists(*sstable_path).ok()) {
      ProcessMegascaleDcn(xspace.get());
      TraceEventsContainer trace_container;
      ConvertXSpaceToTraceEventsContainer(host_name, *xspace, &trace_container);
      std::unique_ptr<tsl::WritableFile> file;
      TF_RETURN_IF_ERROR(
          tsl::Env::Default()->NewWritableFile(*sstable_path, &file));
      TF_RETURN_IF_ERROR(trace_container.StoreAsLevelDbTable(std::move(file)));
    }
    TF_ASSIGN_OR_RETURN(TraceViewOption trace_option,
                        GetTraceViewOption(options));
    auto visibility_filter = std::make_unique<TraceVisibilityFilter>(
        tsl::profiler::MilliSpan(trace_option.start_time_ms,
                                 trace_option.end_time_ms),
        trace_option.resolution);
    TraceEventsContainer trace_container;
    // Trace smaller than threshold will be disabled from streaming.
    constexpr int64_t kDisableStreamingThreshold = 500000;
    TF_RETURN_IF_ERROR(trace_container.LoadFromLevelDbTable(
        *sstable_path, /*filter=*/nullptr, std::move(visibility_filter),
        kDisableStreamingThreshold));
    JsonTraceOptions options;
    IOBufferAdapter adapter(&content);
    TraceEventsToJson<IOBufferAdapter, TraceEventsContainer, RawData>(
        options, trace_container, &adapter);
    return content;
  }
}

absl::StatusOr<std::string> ConvertMultiXSpacesToOverviewPage(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));
  OverviewPage overview_page = ConvertOpStatsToOverviewPage(combined_op_stats);
  InferenceStats inference_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToInferenceStats(session_snapshot, "",
                                                        "", &inference_stats));
  *overview_page.mutable_inference_latency() =
      ComputeInferenceLatencyResult(inference_stats);
  return overview_page.SerializeAsString();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToInputPipeline(
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

absl::StatusOr<std::string> ConvertMultiXSpacesToTfStats(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_kernel_stats_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));
  return ConvertOpStatsToTfStats(combined_op_stats).SerializeAsString();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToKernelStats(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));
  return combined_op_stats.kernel_stats_db().SerializeAsString();
}

absl::StatusOr<std::string> ConvertXSpaceToMemoryProfile(
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

absl::StatusOr<std::string> ConvertMultiXSpacesToPodViewer(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));

  std::string json_output;
  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_primitive_fields = true;
  auto encode_status = tsl::protobuf::util::MessageToJsonString(
      ConvertOpStatsToPodViewer(combined_op_stats), &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return errors::Internal(
        "Could not convert pod viewer to json. Error: ",
        absl::string_view(error_message.data(), error_message.length()));
  }
  return json_output;
}

absl::StatusOr<std::string> ConvertMultiXSpacesToTfDataBottleneckAnalysis(
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

absl::StatusOr<std::string> ConvertMultiXSpacesToHloStats(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, options, &combined_op_stats));
  hlo_stats::HloStatsDatabase hlo_stats_db =
      ConvertOpStatsToHloStats(combined_op_stats);
  return HloStatsToDataTableJson(hlo_stats_db);
}

absl::StatusOr<std::string> ConvertMultiXSpacesToRooflineModel(
    const SessionSnapshot& session_snapshot) {
  OpStatsOptions op_stats_options;
  op_stats_options.generate_op_metrics_db = true;
  OpStats combined_op_stats;
  TF_RETURN_IF_ERROR(ConvertMultiXSpacesToCombinedOpStats(
      session_snapshot, op_stats_options, &combined_op_stats));
  RooflineModelDatabase result =
      ConvertOpStatsToRooflineModel(combined_op_stats, true);
  RooflineModelDatabase result_without_infeed_outfeed =
      ConvertOpStatsToRooflineModel(combined_op_stats, false);
  result.mutable_roofline_model_record()->MergeFrom(
      result_without_infeed_outfeed.roofline_model_record());
  return result.SerializeAsString();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToOpProfileViewer(
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
  tsl::protobuf::util::JsonPrintOptions opts;
  opts.always_print_primitive_fields = true;

  auto encode_status =
      tsl::protobuf::util::MessageToJsonString(profile, &json_output, opts);
  if (!encode_status.ok()) {
    const auto& error_message = encode_status.message();
    return errors::Internal(
        "Could not convert op profile proto to json. Error: ",
        absl::string_view(error_message.data(), error_message.length()));
  }
  return json_output;
}

absl::StatusOr<std::string> PreprocessXSpace(
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

absl::StatusOr<std::string> ConvertDcnCollectiveStatsToToolData(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  // <options> must provide a host_name field.
  std::optional<std::string> hostname =
      GetParam<std::string>(options, "host_name");
  if (!hostname.has_value() || hostname->empty()) {
    return absl::InvalidArgumentError(
        "Cannot find host_name from options for dcn_collective_stats tool.");
  }

  // Load DcnSlackAnalysis for a host.
  TF_ASSIGN_OR_RETURN(
      DcnSlackAnalysis dcnSlackAnalysis,
      GetDcnSlackAnalysisByHostName(session_snapshot, hostname.value()));

  return dcnSlackAnalysis.SerializeAsString();
}

absl::StatusOr<std::string> ConvertMultiXSpacesToInferenceStats(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  InferenceStats inference_stats;
  std::string request_column =
      GetParamWithDefault<std::string>(options, "request_column", "");
  std::string batch_column =
      GetParamWithDefault<std::string>(options, "batch_column", "");
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToInferenceStats(
      session_snapshot, request_column, batch_column, &inference_stats));
  return inference_stats.SerializeAsString();
}

}  // namespace

absl::StatusOr<std::string> ConvertMultiXSpacesToToolData(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
  LOG(INFO) << "serving tool: " << tool_name
            << " with options: " << DebugString(options);
  if (tool_name == "trace_viewer" || tool_name == "trace_viewer@") {
    return ConvertXSpaceToTraceEvents(session_snapshot, tool_name, options);
  } else if (tool_name == "overview_page") {
    return ConvertMultiXSpacesToOverviewPage(session_snapshot);
  } else if (tool_name == "input_pipeline_analyzer") {
    return ConvertMultiXSpacesToInputPipeline(session_snapshot);
  } else if (tool_name == "framework_op_stats") {
    return ConvertMultiXSpacesToTfStats(session_snapshot);
  } else if (tool_name == "kernel_stats") {
    return ConvertMultiXSpacesToKernelStats(session_snapshot);
  } else if (tool_name == "memory_profile") {
    return ConvertXSpaceToMemoryProfile(session_snapshot);
  } else if (tool_name == "pod_viewer") {
    return ConvertMultiXSpacesToPodViewer(session_snapshot);
  } else if (tool_name == "op_profile") {
    return ConvertMultiXSpacesToOpProfileViewer(session_snapshot);
  } else if (tool_name == "hlo_stats") {
    return ConvertMultiXSpacesToHloStats(session_snapshot);
  } else if (tool_name == "roofline_model") {
    return ConvertMultiXSpacesToRooflineModel(session_snapshot);
  } else if (tool_name == "memory_viewer" || tool_name == "graph_viewer") {
    return ConvertHloProtoToToolData(session_snapshot, tool_name, options);
  } else if (tool_name == "tool_names") {
    return GetAvailableToolNames(session_snapshot);
  } else if (tool_name == "_xplane.pb") {  // internal test only.
    return PreprocessXSpace(session_snapshot);
  } else if (tool_name == "inference_profile") {
    return ConvertMultiXSpacesToInferenceStats(session_snapshot, options);
  } else {
    return errors::InvalidArgument(
        "Can not find tool: ", tool_name,
        ". Please update to the latest version of Tensorflow.");
  }
}

}  // namespace profiler
}  // namespace tensorflow
