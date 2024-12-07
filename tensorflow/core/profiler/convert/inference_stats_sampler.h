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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_INFERENCE_STATS_SAMPLER_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_INFERENCE_STATS_SAMPLER_H_

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"

namespace tensorflow::profiler {

// Sampled inference stats of a model.
// The pointers of RequestDetail and BatchDetail point to the actual data stored
// in TfOpStats.InferenceStats.
struct SampledPerModelInferenceStats {
  // Sampled requests and their percentile.
  std::vector<std::pair<const tensorflow::profiler::RequestDetail*, double>>
      sampled_requests;
  // Sampled batches and their percentile.
  std::vector<std::pair<const tensorflow::profiler::BatchDetail*, double>>
      sampled_batches;
};

// All the sampled inference stats of a profile.
// TODO: Move to use SampledInferenceStatsProto if feasible.
using SampledInferenceStats =
    absl::flat_hash_map<int /*model_index*/, SampledPerModelInferenceStats>;

// Samples a subset of InferenceStats from <inference_stats> based on sampling
// column <request_percentile_column> and <batch_percentile_column>.
SampledInferenceStats SampleInferenceStats(
    absl::string_view request_percentile_column,
    absl::string_view batch_percentile_column,
    const tensorflow::profiler::InferenceStats& inference_stats);

}  // namespace tensorflow::profiler

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_INFERENCE_STATS_SAMPLER_H_
