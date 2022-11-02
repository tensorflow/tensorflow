/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_HLO_PROTO_TO_MEMORY_VISUALIZATION_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_HLO_PROTO_TO_MEMORY_VISUALIZATION_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/profiler/protobuf/memory_viewer_preprocess.pb.h"

namespace tensorflow {
namespace profiler {

// Convert HloProto to PreprocessResult proto for memory visualization.
// small_buffer_size sets the byte size within which we collapse buffer entries
// for the max-heap display.
// <heap_simulator_trace_id> is the index of heap simulator trace to be
// displayed. By default it is -1, which means the profiler will infer the heap
// simulator trace id from <memory_color>.
// By default the memory color is 0, which is HBM.
absl::StatusOr<PreprocessResult> ConvertHloProtoToPreprocessResult(
    const xla::HloProto& hlo_proto, int64_t small_buffer_size,
    int64_t heap_simulator_trace_id = -1, int64_t memory_color = 0);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_HLO_PROTO_TO_MEMORY_VISUALIZATION_UTILS_H_
