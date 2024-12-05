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
#include "xla/tsl/profiler/utils/device_utils.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"
#include "tensorflow/core/profiler/convert/inference_stats.h"
#include "tensorflow/core/profiler/convert/inference_stats_combiner.h"
#include "tensorflow/core/profiler/convert/inference_stats_grouping.h"
#include "tensorflow/core/profiler/convert/preprocess_single_host_xplane.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tsl/platform/statusor.h"

namespace tensorflow::profiler {

absl::Status ConvertMultiXSpaceToInferenceStats(
    const SessionSnapshot& session_snapshot, InferenceStats* inference_stats) {
  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                        session_snapshot.GetXSpace(i));
    tsl::profiler::GroupMetadataMap metadata_map;
    StepEvents non_overlapped_step_events;
    InferenceStats inference_stats_per_host;
    std::vector<XPlane*> device_traces =
        tsl::profiler::FindMutableTensorCorePlanes(xspace.get());
    PreprocessSingleHostXSpace(xspace.get(), /*step_grouping=*/true,
                               /*derived_timeline=*/false, &metadata_map);
    GenerateInferenceStats(
        device_traces, non_overlapped_step_events, metadata_map, *xspace,
        tsl::profiler::DeviceType::kTpu, i, &inference_stats_per_host);
    CombineInferenceStatsResult(i, inference_stats_per_host, inference_stats);
  }
  RegroupInferenceStatsByModel(inference_stats);
  return absl::OkStatus();
}
}  // namespace tensorflow::profiler
