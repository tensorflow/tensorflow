// An optimization pass that performs common subexpression elimination.

#ifndef TENSORFLOW_GRAPH_OPTIMIZER_CSE_H_
#define TENSORFLOW_GRAPH_OPTIMIZER_CSE_H_

#include <sys/types.h>
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Perform common-subexpression elimination on the graph "*g".  If
// "consider_fn" is not nullptr, then only nodes for which
// consider_fn(node) returns true will be considered for combining
// during the common subexpression elimination.
extern void OptimizeCSE(Graph* g, std::function<bool(const Node*)> consider_fn);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_OPTIMIZER_CSE_H_
