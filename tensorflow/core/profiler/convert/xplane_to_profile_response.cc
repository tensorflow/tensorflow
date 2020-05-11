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
#include "tensorflow/core/profiler/convert/xplane_to_profile_response.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"
#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"
#include "tensorflow/core/profiler/convert/trace_events_to_json.h"
#include "tensorflow/core/profiler/convert/xplane_to_memory_profile.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kTraceViewer = "trace_viewer";
const absl::string_view kTensorflowStats = "tensorflow_stats";
const absl::string_view kInputPipeline = "input_pipeline";
const absl::string_view kOverviewPage = "overview_page";
const absl::string_view kKernelStats = "kernel_stats";
const absl::string_view kMemoryProfile = "memory_profile";
const absl::string_view kXPlane = "xplane";

HardwareType HardwareTypeFromRunEnvironment(const RunEnvironment& run_env) {
  if (run_env.device_type() == "GPU") return HardwareType::GPU;
  if (run_env.device_type() == "CPU") return HardwareType::CPU_ONLY;
  return HardwareType::UNKNOWN_HARDWARE;
}

template <typename Proto>
void AddToolData(absl::string_view tool_name, const Proto& tool_output,
                 ProfileResponse* response) {
  auto* tool_data = response->add_tool_data();
  tool_data->set_name(string(tool_name));
  tool_output.SerializeToString(tool_data->mutable_data());
}

template <typename Proto>
Status ConvertProtoToJson(const Proto& proto_output, std::string* json_output) {
  protobuf::util::JsonPrintOptions json_options;
  json_options.always_print_primitive_fields = true;
  auto status = protobuf::util::MessageToJsonString(proto_output, json_output,
                                                    json_options);
  if (!status.ok()) {
    // Convert error_msg google::protobuf::StringPiece (or absl::string_view) to
    // tensorflow::StringPiece.
    auto error_msg = status.message();
    return errors::Internal(
        strings::StrCat("Could not convert proto to JSON string: ",
                        StringPiece(error_msg.data(), error_msg.length())));
  }
  return Status::OK();
}

// Returns the tool name with extension.
string ToolName(absl::string_view tool) {
  if (tool == kTraceViewer) return "trace.json.gz";
  if (tool == kMemoryProfile) return "memory_profile.json.gz";
  return absl::StrCat(tool, ".pb");
}

}  // namespace

Status ConvertXSpaceToProfileResponse(const XSpace& xspace,
                                      const ProfileRequest& req,
                                      ProfileResponse* response) {
  absl::flat_hash_set<absl::string_view> tools(req.tools().begin(),
                                               req.tools().end());
  AddToolData(ToolName(kXPlane), xspace, response);
  if (tools.empty()) return Status::OK();
  if (tools.contains(kTraceViewer)) {
    Trace trace;
    ConvertXSpaceToTraceEvents(xspace, &trace);
    if (trace.trace_events().empty()) {
      response->set_empty_trace(true);
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(SaveGzippedToolDataToTensorboardProfile(
        req.repository_root(), req.session_id(), req.host_name(),
        ToolName(kTraceViewer), TraceEventsToJson(trace)));
    // Trace viewer is the only tool, skip OpStats conversion.
    if (tools.size() == 1) return Status::OK();
  }
  OpStats op_stats = ConvertXSpaceToOpStats(xspace);
  HardwareType hw_type =
      HardwareTypeFromRunEnvironment(op_stats.run_environment());
  if (tools.contains(kOverviewPage)) {
    OverviewPage overview_page_db =
        ConvertOpStatsToOverviewPage(op_stats, hw_type);
    AddToolData(ToolName(kOverviewPage), overview_page_db, response);
    if (tools.contains(kInputPipeline)) {
      AddToolData(ToolName(kInputPipeline), overview_page_db.input_analysis(),
                  response);
    }
  } else if (tools.contains(kInputPipeline)) {
    InputPipelineAnalysisResult input_pipeline_analysis =
        ConvertOpStatsToInputPipelineAnalysis(op_stats, hw_type);
    AddToolData(ToolName(kInputPipeline), input_pipeline_analysis, response);
  }
  if (tools.contains(kTensorflowStats)) {
    TfStatsDatabase tf_stats_db = ConvertOpStatsToTfStats(op_stats);
    AddToolData(ToolName(kTensorflowStats), tf_stats_db, response);
  }
  if (tools.contains(kKernelStats)) {
    AddToolData(ToolName(kKernelStats), op_stats.kernel_stats_db(), response);
  }
  if (tools.contains(kMemoryProfile)) {
    if (const XPlane* host_plane = FindPlaneWithName(xspace, kHostThreads)) {
      MemoryProfile memory_profile = ConvertXPlaneToMemoryProfile(*host_plane);
      std::string json_output;
      TF_RETURN_IF_ERROR(ConvertProtoToJson(memory_profile, &json_output));
      TF_RETURN_IF_ERROR(SaveGzippedToolDataToTensorboardProfile(
          req.repository_root(), req.session_id(), req.host_name(),
          ToolName(kMemoryProfile), json_output));
    }
  }
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
