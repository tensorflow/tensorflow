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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_INFERENCE_STATS_COMBINER_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_INFERENCE_STATS_COMBINER_H_
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"

namespace tensorflow::profiler {
void CombineInferenceStatsResult(int src_host_id, const InferenceStats& src,
                                 InferenceStats* dst);
}  // namespace tensorflow::profiler

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_INFERENCE_STATS_COMBINER_H_
