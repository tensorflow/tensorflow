#ifndef TENSORFLOW_GRAPH_EQUAL_GRAPH_DEF_H_
#define TENSORFLOW_GRAPH_EQUAL_GRAPH_DEF_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

// Determines if actual and expected are equal, ignoring ordering of
// nodes, attrs, and control inputs.  If the GraphDefs are different
// and diff != nullptr, *diff is set to an explanation of the
// difference.  Note that we use node names to match up nodes between
// the graphs, and so the naming of nodes must be consistent.
bool EqualGraphDef(const GraphDef& actual, const GraphDef& expected,
                   string* diff);

// Determines if actual and expected are equal, ignoring ordering of
// attrs and control inputs.  If the NodeDefs are different and
// diff != nullptr, *diff is set to an explanation of the difference.
bool EqualNodeDef(const NodeDef& actual, const NodeDef& expected, string* diff);

#define TF_EXPECT_GRAPH_EQ(expected, actual)                  \
  do {                                                        \
    string diff;                                              \
    EXPECT_TRUE(EqualGraphDef(actual, expected, &diff))       \
        << diff << "\nActual: " << SummarizeGraphDef(actual); \
  } while (false)

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_EQUAL_GRAPH_DEF_H_
