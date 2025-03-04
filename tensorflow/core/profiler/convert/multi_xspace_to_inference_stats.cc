/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/convert/multi_xspace_to_inference_stats.h"

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/device_utils.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/convert/inference_stats.h"
#include "tensorflow/core/profiler/convert/inference_stats_combiner.h"
#include "tensorflow/core/profiler/convert/inference_stats_grouping.h"
#include "tensorflow/core/profiler/convert/inference_stats_sampler.h"
#include "tensorflow/core/profiler/convert/preprocess_single_host_xplane.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/convert/xplane_to_step_events.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow::profiler {

namespace {
using tsl::profiler::FindMutablePlanesWithPrefix;
using tsl::profiler::FindMutablePlaneWithName;

SampledInferenceStatsProto GetSampledInferenceStatsProto(
    const InferenceStats& inference_stats, absl::string_view request_column,
    absl::string_view batch_column) {
  SampledInferenceStatsProto result;
  SampledInferenceStats sampled_stats =
      SampleInferenceStats(request_column, batch_column, inference_stats);
  for (const auto& [model_index, samples] : sampled_stats) {
    SampledPerModelInferenceStatsProto per_model_stats;
    for (const auto& [request, percentile] : samples.sampled_requests) {
      RequestDetail request_detail = *request;
      request_detail.set_percentile(percentile);
      *per_model_stats.add_sampled_requests() = request_detail;
    }
    for (const auto& [batch, percentile] : samples.sampled_batches) {
      BatchDetail batch_detail = *batch;
      batch_detail.set_percentile(percentile);
      *per_model_stats.add_sampled_batches() = batch_detail;
    }
    result.mutable_sampled_inference_stats_per_model()->insert(
        {model_index, per_model_stats});
  }
  return result;
}
}  // namespace

StepEvents GetNonOverlappedStepEvents(XSpace* xspace) {
  StepEvents non_overlapped_step_events;

  std::vector<XPlane*> device_traces =
      FindMutablePlanesWithPrefix(xspace, kGpuPlanePrefix);
  if (device_traces.empty()) return non_overlapped_step_events;

  StepEvents device_step_events;
  StepEvents host_step_events;
  for (XPlane* device_trace : device_traces) {
    StepEvents events = ConvertDeviceTraceXPlaneToStepEvents(*device_trace);
    UnionCombineStepEvents(events, &device_step_events);
  }

  XPlaneVisitor host_plane = tsl::profiler::CreateTfXPlaneVisitor(
      FindMutablePlaneWithName(xspace, kHostThreadsPlaneName));

  host_plane.ForEachLine([&](const XLineVisitor& line) {
    StepEvents events =
        ConvertHostThreadsXLineToStepEvents(line, &device_step_events);
    UnionCombineStepEvents(events, &host_step_events);
  });
  StepEvents overlapped_step_events;
  UnionCombineStepEvents(device_step_events, &overlapped_step_events);
  UnionCombineStepEvents(host_step_events, &overlapped_step_events);
  non_overlapped_step_events =
      ToNonOverlappedStepEvents(overlapped_step_events);
  return non_overlapped_step_events;
}

absl::Status ConvertMultiXSpaceToInferenceStats(
    const SessionSnapshot& session_snapshot, absl::string_view request_column,
    absl::string_view batch_column, InferenceStats* inference_stats) {
  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                        session_snapshot.GetXSpace(i));
    tsl::profiler::GroupMetadataMap metadata_map;
    InferenceStats inference_stats_per_host;
    std::vector<XPlane*> device_traces =
        tsl::profiler::FindMutableTensorCorePlanes(xspace.get());
    PreprocessSingleHostXSpace(xspace.get(), /*step_grouping=*/true,
                               /*derived_timeline=*/false, &metadata_map);
    StepEvents non_overlapped_step_events =
        GetNonOverlappedStepEvents(xspace.get());
    GenerateInferenceStats(
        device_traces, non_overlapped_step_events, metadata_map, *xspace,
        tsl::profiler::DeviceType::kTpu, i, &inference_stats_per_host);
    CombineInferenceStatsResult(i, inference_stats_per_host, inference_stats);
  }
  RegroupInferenceStatsByModel(inference_stats);
  *inference_stats->mutable_sampled_inference_stats() =
      GetSampledInferenceStatsProto(*inference_stats, request_column,
                                    batch_column);
  return absl::OkStatus();
}
}  // namespace tensorflow::profiler
