#include "tensorflow/core/graph/costutil.h"

#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/costmodel.h"

namespace tensorflow {

std::vector<int64> LongestOutgoingPathCost(const Graph& graph,
                                           const CostModel& cm) {
  std::vector<int64> result(graph.num_node_ids());
  DFS(graph, nullptr, [&result, &cm](Node* n) {
    int64 max_child = 0;
    for (const Node* out : n->out_nodes()) {
      max_child = std::max(max_child, result[out->id()]);
    }
    result[n->id()] = max_child + (n->IsOp() ? cm.TimeEstimate(n).value() : 0);
  });
  return result;
}

}  // namespace tensorflow
