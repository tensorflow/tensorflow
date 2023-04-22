/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_SEGMENT_SEGMENT_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_SEGMENT_SEGMENT_H_

#include <set>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2tensorrt/segment/union_find.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace segment {

constexpr char kTftrtOpMaxBatchSizeAttr[] = "_tftrt_op_max_batch_size";

struct SegmentOptions {
  // This struct holds per graph segmenting parameters.
  // Segment must contain at least this many nodes.
  int minimum_segment_size = 2;
  bool use_implicit_batch = true;
  // The maximum batch size used to build the engines in the graph, when
  // use_implicit_batch is true.
  absl::optional<int> maximum_batch_size = absl::nullopt;
  // When use_implicit_batch is false or when we are building dynamic engines,
  // we allow dynamic non-batch dimensions.
  bool allow_dynamic_non_batch_dim = false;
  // The name of the device to put the segment on.
  std::set<string> exclude_node_list;
};

struct NodePtrCompare {
  bool operator()(const Node* lhs, const Node* rhs) const {
    return lhs->name() < rhs->name();
  }
};

struct Segment {
  Segment() {}
  Segment(const ClusterProperty& property,
          const std::set<const Node*, NodePtrCompare>& nodes)
      : property(property), nodes(nodes) {}
  ClusterProperty property;
  std::set<const Node*, NodePtrCompare> nodes;
};

// Vector of segments, each entry contains a set of node pointers.
using SegmentVector = std::vector<Segment>;

// Get the subgraphs of a graph that can be handled by TensorRT.
//
// @param tf_graph Graph of the network.
// @graph_properties is the static graph properties.
// @param candidate_fn A function that returns OK for a Node* if
// that node can be handled by TensorRT.
// @param segments Returns the TensorRT segments/subgraphs. Each entry
// in the vector describes a subgraph by giving a set of the names of
// all the NodeDefs in that subgraph.
// @return the status.
Status SegmentGraph(const Graph* tf_graph,
                    const grappler::GraphProperties* graph_properties,
                    const std::function<Status(const Node*)>& candidate_fn,
                    const std::function<bool(const Edge*)>& input_candidate_fn,
                    const std::function<bool(const Edge*)>& output_candidate_fn,
                    const SegmentOptions& options, SegmentVector* segments);

}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_SEGMENT_SEGMENT_H_
