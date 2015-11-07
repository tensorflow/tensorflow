#ifndef TENSORFLOW_GRAPH_COSTUTIL_H_
#define TENSORFLOW_GRAPH_COSTUTIL_H_

#include <vector>
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

class CostModel;
class Graph;

// result[i] is an estimate of the longest execution path from
// the node with id i to the sink node.
std::vector<int64> LongestOutgoingPathCost(const Graph& graph,
                                           const CostModel& cm);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_COSTUTIL_H_
