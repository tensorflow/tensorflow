/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_STREAM_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_STREAM_UTIL_H_

#include <unordered_map>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace gpu_stream_util {

struct AssignStreamsOpts {
  int32 max_streams = 1;
  // The following options specify a stream to use for specific op
  // types.  The value -1 allows ops to be assigned to any stream.
  int32 send_stream = -1;
  int32 recv_stream = -1;
  int32 const_stream = -1;
  int32 compute_stream = -1;
};

// Given the input graph, assigns every node in the graph with a
// stream_id that should be used.
Status AssignStreams(const Graph* graph, const AssignStreamsOpts& opts,
                     std::unordered_map<int, int>* node_to_stream_id);

}  // namespace gpu_stream_util
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_STREAM_UTIL_H_
