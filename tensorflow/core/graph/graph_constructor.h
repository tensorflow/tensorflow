#ifndef TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_
#define TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

// Construct a graph *g out of a GraphDef gdef. Returns non-OK on
// error, in which case *g is left in an incomplete state.
struct GraphConstructorOptions {
  // If true, allows internal ops in the GraphDef.
  bool allow_internal_ops = false;

  // If true, the graph def is expected to have fully specified
  // devices for all nodes. A node in the resulting graph "g" has the
  // device name set accordingly.
  //
  // TODO(zhifengc): if possible, consider removing this option.
  bool expect_device_spec = false;

  // If true, perform common subexpression elimination on the graph.
  // TODO(jeff): Turn this default to true?
  bool optimizer_do_cse = false;

  // If "optimizer_do_cse" is true and "cse_consider_function" is
  // not nullptr, then only consider nodes for CSE for which
  // "cse_consider_function(node)" returns true.
  std::function<bool(const Node*)> cse_consider_function = nullptr;
};
extern Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
                                          const GraphDef& gdef, Graph* g);

// Make a copy of "src" into "*dest".
//
// REQUIRES: "*dest" is a freshly allocated graph without any nodes or edges
// other than the implicit Source/Sink nodes.
extern void CopyGraph(const Graph& src, Graph* dest);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_
