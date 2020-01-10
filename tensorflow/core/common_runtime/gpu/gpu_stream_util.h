#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_STREAM_UTIL_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_STREAM_UTIL_H_

#include <unordered_map>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/status.h"

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

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_STREAM_UTIL_H_
