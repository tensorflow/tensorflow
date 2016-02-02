/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/graph_def_util.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/graph/equal_graph_def.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Producer and consumer have default for an attr -> graph unchanged.
TEST(RemoveNewDefaultAttrsFromGraphDefTest, NoChangeWithDefault) {
  OpList op_list;
  TF_ASSERT_OK(OpDefBuilder("NoChangeWithDefault")
                   .Attr("a: int = 12")
                   .Finalize(op_list.add_op()));
  OpListOpRegistry registry(&op_list);

  GraphDef graph_def;
  TF_ASSERT_OK(NodeDefBuilder("ncwd", "NoChangeWithDefault", &registry)
                   .Finalize(graph_def.add_node()));
  GraphDef expected_graph_def = graph_def;

  std::set<std::pair<string, string>> op_attr_removed;
  TF_ASSERT_OK(RemoveNewDefaultAttrsFromGraphDef(&graph_def, registry, registry,
                                                 &op_attr_removed));

  TF_EXPECT_GRAPH_EQ(expected_graph_def, graph_def);
  EXPECT_TRUE(op_attr_removed.empty());
}

// Producer and consumer both have an attr -> graph unchanged.
TEST(RemoveNewDefaultAttrsFromGraphDefTest, NoChangeNoDefault) {
  OpList op_list;
  TF_ASSERT_OK(OpDefBuilder("NoChangeNoDefault")
                   .Attr("a: int")
                   .Finalize(op_list.add_op()));
  OpListOpRegistry registry(&op_list);

  GraphDef graph_def;
  TF_ASSERT_OK(NodeDefBuilder("ncnd", "NoChangeNoDefault", &registry)
                   .Attr("a", 42)
                   .Finalize(graph_def.add_node()));
  GraphDef expected_graph_def = graph_def;

  std::set<std::pair<string, string>> op_attr_removed;
  TF_ASSERT_OK(RemoveNewDefaultAttrsFromGraphDef(&graph_def, registry, registry,
                                                 &op_attr_removed));

  TF_EXPECT_GRAPH_EQ(expected_graph_def, graph_def);
  EXPECT_TRUE(op_attr_removed.empty());
}

// Producer has default for an attr that the consumer does not know
// about, and the produced graph has the default value for the attr ->
// attr removed from graph (and so able to be consumed).
TEST(RemoveNewDefaultAttrsFromGraphDefTest, UsesDefault) {
  OpList consumer_op_list;
  TF_ASSERT_OK(OpDefBuilder("UsesDefault").Finalize(consumer_op_list.add_op()));
  OpListOpRegistry consumer_registry(&consumer_op_list);

  OpList producer_op_list;
  TF_ASSERT_OK(OpDefBuilder("UsesDefault")
                   .Attr("a: int = 17")
                   .Finalize(producer_op_list.add_op()));
  OpListOpRegistry producer_registry(&producer_op_list);

  GraphDef produced_graph_def;
  TF_ASSERT_OK(NodeDefBuilder("uses_default", "UsesDefault", &producer_registry)
                   .Finalize(produced_graph_def.add_node()));

  std::set<std::pair<string, string>> op_attr_removed;
  TF_ASSERT_OK(
      RemoveNewDefaultAttrsFromGraphDef(&produced_graph_def, consumer_registry,
                                        producer_registry, &op_attr_removed));

  GraphDef expected_graph_def;
  TF_ASSERT_OK(NodeDefBuilder("uses_default", "UsesDefault", &consumer_registry)
                   .Finalize(expected_graph_def.add_node()));
  TF_EXPECT_GRAPH_EQ(expected_graph_def, produced_graph_def);

  std::set<std::pair<string, string>> expected_removed({{"UsesDefault", "a"}});
  EXPECT_EQ(expected_removed, op_attr_removed);
}

// Producer has default for an attr that the consumer does not know
// about, graph sets the attr to a value different from the default ->
// graph unchanged (but not able to be consumed by consumer).
TEST(RemoveNewDefaultAttrsFromGraphDefTest, ChangedFromDefault) {
  OpList consumer_op_list;
  TF_ASSERT_OK(
      OpDefBuilder("ChangedFromDefault").Finalize(consumer_op_list.add_op()));
  OpListOpRegistry consumer_registry(&consumer_op_list);

  OpList producer_op_list;
  TF_ASSERT_OK(OpDefBuilder("ChangedFromDefault")
                   .Attr("a: int = 17")
                   .Finalize(producer_op_list.add_op()));
  OpListOpRegistry producer_registry(&producer_op_list);

  GraphDef produced_graph_def;
  TF_ASSERT_OK(NodeDefBuilder("changed_from_default", "ChangedFromDefault",
                              &producer_registry)
                   .Attr("a", 9)
                   .Finalize(produced_graph_def.add_node()));
  GraphDef expected_graph_def = produced_graph_def;

  std::set<std::pair<string, string>> op_attr_removed;
  TF_ASSERT_OK(
      RemoveNewDefaultAttrsFromGraphDef(&produced_graph_def, consumer_registry,
                                        producer_registry, &op_attr_removed));

  TF_EXPECT_GRAPH_EQ(expected_graph_def, produced_graph_def);
  EXPECT_TRUE(op_attr_removed.empty());
}

}  // namespace
}  // namespace tensorflow
