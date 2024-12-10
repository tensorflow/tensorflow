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

#include "tensorflow/core/profiler/convert/compute_inference_latency.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include "absl/log/log.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"

namespace tensorflow::profiler {

struct LatencyBreakdown {
  double total_latency_us = 0.0;
  double host_latency_us = 0.0;
  double device_latency_us = 0.0;
  double communication_latency_us = 0.0;
};

void SetLatencyBreakdown(const LatencyBreakdown& src,
                         OverviewLatencyBreakdown* res) {
  res->set_total_latency_us(src.total_latency_us);
  res->set_host_latency_us(src.host_latency_us);
  res->set_device_latency_us(src.device_latency_us);
  res->set_communication_latency_us(src.communication_latency_us);
}

void SafeDivide(int64_t count, double* num) {
  constexpr double kEpsilon = 1.0e-20;
  if (count == 0 || std::abs(*num) < kEpsilon) {
    *num = 0.0;
  } else {
    *num /= count;
  }
}

void ComputeAverage(int64_t count, LatencyBreakdown* breakdown) {
  SafeDivide(count, &breakdown->total_latency_us);
  SafeDivide(count, &breakdown->host_latency_us);
  SafeDivide(count, &breakdown->device_latency_us);
  SafeDivide(count, &breakdown->communication_latency_us);
}

void ComputeBreakdownFromSessionRun(
    const tensorflow::profiler::RequestDetail& request_detail,
    LatencyBreakdown* res, LatencyBreakdown* avg) {
  double session_run_duration_us = tsl::profiler::PicoToMicro(
      request_detail.end_time_ps() - request_detail.start_time_ps());
  double device_time_us =
      tsl::profiler::PicoToMicro(request_detail.device_time_ps());
  double communication_time_us =
      tsl::profiler::PicoToMicro(request_detail.read_from_device_time_ps() +
                                 request_detail.write_to_device_time_ps());
  double host_time_us =
      session_run_duration_us - device_time_us - communication_time_us;
  *res = {session_run_duration_us, host_time_us, device_time_us,
          communication_time_us};

  avg->total_latency_us += session_run_duration_us;
  avg->device_latency_us += device_time_us;
  avg->communication_latency_us += communication_time_us;
  avg->host_latency_us +=
      session_run_duration_us - device_time_us - communication_time_us;
}

// Compute the inference latency from inference stats proto.
OverviewInferenceLatency ComputeInferenceLatencyResult(
    const tensorflow::profiler::InferenceStats& inference_stats) {
  OverviewInferenceLatency result;
  // If inference_stats is empty, return early with empty result.
  // The following code is able to return empty result even
  // without early return.
  if (inference_stats.inference_stats_per_model_size() == 0) return result;

  // Target percentiles over all session runs.
  // Default is [50.0, 75.0, 90.0, 99.0, 99.9].
  constexpr double kTargetPercentiles[] = {50.0, 75.0, 90.0, 99.0, 99.9};
  // Saves the latency corresponding to each percentile.

  std::vector<LatencyBreakdown> sessions;
  double total_sessioins_per_sec = 0;
  double max_latency = 0.0;
  double min_latency = std::numeric_limits<double>::max();
  LatencyBreakdown avg;
  // Iterate over all session runs from all models, calculate the device,
  // communication, and host time for each session run, and push in the
  // vector<LatencyBreakdown> sessions. Also update the max, min, count, avg.
  for (const auto& model_inference_stats :
       inference_stats.inference_stats_per_model()) {
    total_sessioins_per_sec +=
        model_inference_stats.second.request_throughput();
    for (const auto& request_detail :
         model_inference_stats.second.request_details()) {
      LatencyBreakdown session_breakdown;
      ComputeBreakdownFromSessionRun(request_detail, &session_breakdown, &avg);
      sessions.push_back(session_breakdown);
      double session_run_duration_us = tsl::profiler::PicoToMicro(
          request_detail.end_time_ps() - request_detail.start_time_ps());
      max_latency = std::max(max_latency, session_run_duration_us);
      min_latency = std::min(min_latency, session_run_duration_us);
    }
  }
  // Return empty result if there is no session found.
  if (sessions.empty()) return result;
  result.set_sessions_per_second(total_sessioins_per_sec);
  result.set_max_latency_us(max_latency);
  result.set_min_latency_us(min_latency);
  ComputeAverage(sessions.size(), &avg);

  // Sort the sessions based on session run duration. For a specified
  // percentile, get the corresponding session with the (lower-bound) index.
  std::sort(sessions.begin(), sessions.end(),
            [](const LatencyBreakdown& a, const LatencyBreakdown& b) {
              return a.total_latency_us < b.total_latency_us;
            });
  for (const auto& percent : kTargetPercentiles) {
    result.add_percentile_numbers(percent);
    int64_t index = percent / 100.0 * sessions.size();
    SetLatencyBreakdown(sessions[index], result.add_latency_breakdowns());
  }
  // Set the average latency stats.
  SetLatencyBreakdown(avg, result.add_latency_breakdowns());

  return result;
}

}  // namespace tensorflow::profiler
