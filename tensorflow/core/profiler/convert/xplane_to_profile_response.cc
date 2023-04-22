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

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"
#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"
#include "tensorflow/core/profiler/convert/trace_events_to_json.h"
#include "tensorflow/core/profiler/convert/xplane_to_memory_profile.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kTraceViewer = "trace_viewer";
const absl::string_view kTensorflowStats = "tensorflow_stats";
const absl::string_view kInputPipeline = "input_pipeline";
const absl::string_view kOverviewPage = "overview_page";
const absl::string_view kKernelStats = "kernel_stats";
const absl::string_view kMemoryProfile = "memory_profile";
const absl::string_view kXPlanePb = "xplane.pb";

template <typename Proto>
void AddToolData(absl::string_view tool_name, const Proto& tool_output,
                 ProfileResponse* response) {
  auto* tool_data = response->add_tool_data();
  tool_data->set_name(string(tool_name));
  tool_output.SerializeToString(tool_data->mutable_data());
}

// Returns the tool name with extension.
std::string ToolName(absl::string_view tool) {
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
  if (tools.empty()) return Status::OK();
  if (tools.contains(kXPlanePb)) {
    AddToolData(kXPlanePb, xspace, response);
  }
  if (tools.contains(kTraceViewer)) {
    Trace trace;
    ConvertXSpaceToTraceEvents(xspace, &trace);
    if (trace.trace_events().empty()) {
      response->set_empty_trace(true);
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(SaveGzippedToolData(
        req.repository_root(), req.session_id(), req.host_name(),
        ToolName(kTraceViewer), TraceEventsToJson(trace)));
    // Trace viewer is the only tool, skip OpStats conversion.
    if (tools.size() == 1) return Status::OK();
  }

  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  options.maybe_drop_incomplete_steps = true;
  OpStats op_stats = ConvertXSpaceToOpStats(xspace, options);
  if (tools.contains(kOverviewPage)) {
    OverviewPage overview_page_db = ConvertOpStatsToOverviewPage(op_stats);
    AddToolData(ToolName(kOverviewPage), overview_page_db, response);
    if (tools.contains(kInputPipeline)) {
      AddToolData(ToolName(kInputPipeline), overview_page_db.input_analysis(),
                  response);
    }
  } else if (tools.contains(kInputPipeline)) {
    AddToolData(ToolName(kInputPipeline),
                ConvertOpStatsToInputPipelineAnalysis(op_stats), response);
  }
  if (tools.contains(kTensorflowStats)) {
    TfStatsDatabase tf_stats_db = ConvertOpStatsToTfStats(op_stats);
    AddToolData(ToolName(kTensorflowStats), tf_stats_db, response);
  }
  if (tools.contains(kKernelStats)) {
    AddToolData(ToolName(kKernelStats), op_stats.kernel_stats_db(), response);
  }
  if (tools.contains(kMemoryProfile)) {
    std::string json_output;
    TF_RETURN_IF_ERROR(ConvertXSpaceToMemoryProfileJson(xspace, &json_output));
    TF_RETURN_IF_ERROR(SaveGzippedToolData(
        req.repository_root(), req.session_id(), req.host_name(),
        ToolName(kMemoryProfile), json_output));
  }
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
