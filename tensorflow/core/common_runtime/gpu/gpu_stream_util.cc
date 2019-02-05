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

#include "tensorflow/core/common_runtime/gpu/gpu_stream_util.h"

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace gpu_stream_util {

Status AssignStreams(const Graph* graph, const AssignStreamsOpts& opts,
                     std::unordered_map<int, int>* node_to_stream_id) {
  VLOG(1) << "AssignStreams";
  Status status;

  // Sanity check arguments.
  if (graph == nullptr)
    status.Update(errors::InvalidArgument("Bad graph argument supplied."));
  if (node_to_stream_id == nullptr) {
    status.Update(
        errors::InvalidArgument("Bad node_to_stream_id argument supplied."));
  }
  if ((opts.max_streams < 1) || (opts.send_stream >= opts.max_streams) ||
      (opts.recv_stream >= opts.max_streams) ||
      (opts.const_stream >= opts.max_streams) ||
      (opts.compute_stream >= opts.max_streams)) {
    status.Update(errors::InvalidArgument("Bad graph argument supplied."));
  }
  TF_RETURN_IF_ERROR(status);

  // Topologically sort the nodes.
  std::vector<Node*> order;
  GetReversePostOrder(*graph, &order);
  if (VLOG_IS_ON(2)) {
    for (Node* n : order) {
      const int node_id = n->id();
      VLOG(2) << "Node " << node_id << " " << n->type_string() << " "
              << n->name() << " " << n->in_edges().size() << " inputs";
      for (const Edge* e : n->in_edges()) {
        VLOG(2) << "  Edge from " << e->src()->id() << "  " << e->src()->name()
                << " fanout " << e->src()->out_edges().size();
      }
    }
  }
  // We perform stream assignment assuming a large number of
  // stream IDs and then map these down to the required number of streams
  // using simple round-robin.
  // Stream Assignment strategy:
  // 1. Nodes with zero inputs are always be executed on a
  // fresh stream.
  // 2. Try to execute a node on the same stream as one of its
  // inputs to avoid inter-stream dependencies.
  // 3. If any input comes from a node with a large fanout then
  // perhaps an indication that it is shared between parallel
  // streams of work. We choose a new stream here so that all consumers
  // of the tensor are likely to run in parallel.
  int highest_stream_id = -1;
  for (Node* n : order) {
    VLOG(3) << "Inspecting node " << n->DebugString();
    const int node_id = n->id();
    const string& op = n->type_string();

    // Determine a suitable stream to use.
    int stream_id = highest_stream_id + 1;
    for (const Edge* e : n->in_edges()) {
      const size_t fanout = e->src()->out_edges().size();
      if (fanout == 1) {
        stream_id = (*node_to_stream_id)[e->src()->id()];
        break;
      }
    }
    // Override stream for specific op types.
    if (op == "_Send") {
      if (opts.send_stream >= 0) stream_id = opts.send_stream;
    } else if (op == "_Recv") {
      if (opts.recv_stream >= 0) stream_id = opts.recv_stream;
    } else if (op == "Const") {
      if (opts.const_stream >= 0) stream_id = opts.const_stream;
    } else {
      if (opts.compute_stream >= 0) stream_id = opts.compute_stream;
    }

    (*node_to_stream_id)[node_id] = stream_id % opts.max_streams;
    highest_stream_id = std::max(stream_id, highest_stream_id);
  }
  VLOG(1) << "Identified " << highest_stream_id << " candidate streams for "
          << order.size() << " nodes.";

  return Status::OK();
}

}  // namespace gpu_stream_util
}  // namespace tensorflow
