#include "tensorflow/core/graph/node_builder.h"

#include <gtest/gtest.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

REGISTER_OP("Source").Output("o: out_types").Attr("out_types: list(type)");
REGISTER_OP("Sink").Input("i: T").Attr("T: type");

TEST(NodeBuilderTest, Simple) {
  RequireDefaultOps();
  Graph graph(OpRegistry::Global());
  Node* source_node;
  EXPECT_OK(NodeBuilder("source_op", "Source")
                .Attr("out_types", {DT_INT32, DT_STRING})
                .Finalize(&graph, &source_node));
  ASSERT_TRUE(source_node != nullptr);

  // Try connecting to each of source_node's outputs.
  EXPECT_OK(NodeBuilder("sink1", "Sink")
                .Input(source_node)
                .Finalize(&graph, nullptr));
  EXPECT_OK(NodeBuilder("sink2", "Sink")
                .Input(source_node, 1)
                .Finalize(&graph, nullptr));

  // Generate an error if the index is out of range.
  EXPECT_FALSE(NodeBuilder("sink3", "Sink")
                   .Input(source_node, 2)
                   .Finalize(&graph, nullptr)
                   .ok());
  EXPECT_FALSE(NodeBuilder("sink4", "Sink")
                   .Input(source_node, -1)
                   .Finalize(&graph, nullptr)
                   .ok());
  EXPECT_FALSE(NodeBuilder("sink5", "Sink")
                   .Input({source_node, -1})
                   .Finalize(&graph, nullptr)
                   .ok());

  // Generate an error if the node is nullptr.  This can happen when using
  // GraphDefBuilder if there was an error creating the input node.
  EXPECT_FALSE(NodeBuilder("sink6", "Sink")
                   .Input(nullptr)
                   .Finalize(&graph, nullptr)
                   .ok());
  EXPECT_FALSE(NodeBuilder("sink7", "Sink")
                   .Input(NodeBuilder::NodeOut(nullptr, 0))
                   .Finalize(&graph, nullptr)
                   .ok());
}

}  // namespace
}  // namespace tensorflow
