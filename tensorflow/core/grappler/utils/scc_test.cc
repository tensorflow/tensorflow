/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/scc.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class SCCTest : public ::testing::Test {
 public:
  void SetUp() override {
    std::unordered_map<string, DeviceProperties> devices;
    DeviceProperties unknown_device;
    devices["MY_DEVICE"] = unknown_device;
    cluster_.reset(new VirtualCluster(devices));
    TF_CHECK_OK(cluster_->Provision());
  }

  void TearDown() override { cluster_.reset(); }

 protected:
  static NodeDef CreateNode(const string& name,
                            gtl::ArraySlice<string> inputs) {
    NodeDef node;
    node.set_name(name);
    for (const string& input : inputs) {
      node.add_input(input);
    }
    return node;
  }

  std::unique_ptr<VirtualCluster> cluster_;
};

TEST_F(SCCTest, NoLoops) {
  // Create a simple graph without any loop.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  std::unordered_map<const NodeDef*, int> components;
  int num_components;
  StronglyConnectedComponents(item.graph, &components, &num_components);

  EXPECT_EQ(num_components, 1);
  for (const auto& node : item.graph.node()) {
    EXPECT_EQ(-1, components[&node]);
  }
}

TEST_F(SCCTest, DisjointCycleAndPath) {
  GraphDef graph;
  // Create a cycle
  *graph.add_node() = CreateNode("a", {"d"});
  *graph.add_node() = CreateNode("b", {"a"});
  *graph.add_node() = CreateNode("c", {"b"});
  *graph.add_node() = CreateNode("d", {"c"});

  // Add a path disjoint from cycle
  *graph.add_node() = CreateNode("e", {});
  *graph.add_node() = CreateNode("f", {"e"});
  *graph.add_node() = CreateNode("g", {"f"});
  *graph.add_node() = CreateNode("h", {"g"});

  std::vector<const NodeDef*> nodes;
  std::unordered_map<string, const NodeDef*> name_to_node;
  for (const auto& n : graph.node()) {
    nodes.push_back(&n);
    name_to_node[n.name()] = &n;
  }

  int num_components;
  std::unordered_map<const NodeDef*, int> components;
  StronglyConnectedComponents(graph, &components, &num_components);

  EXPECT_EQ(num_components, 2);

  for (const auto& pair : {std::make_pair("a", "b"), std::make_pair("a", "c"),
                           std::make_pair("a", "d")}) {
    EXPECT_EQ(components[name_to_node[pair.first]],
              components[name_to_node[pair.second]]);
  }

  for (const auto& node : {"e", "f", "g", "h"})
    EXPECT_EQ(-1, components[name_to_node[node]]);
}
}  // namespace

TEST_F(SCCTest, WikipediaExample) {
  // Graph with 4 SCCs:

  // SCC1:
  // a -> b
  // b -> c
  // c -> a

  // d -> b
  // d -> c

  // SCC2:
  // d -> e
  // e -> d

  // e -> f
  // f -> c

  // SCC3:
  // f -> g
  // g -> f

  // h -> g
  // h -> d

  // SCC4:
  // h -> h

  // NodeDefs define inbound connections (inputs)
  GraphDef graph;
  *graph.add_node() = CreateNode("a", {"c"});
  *graph.add_node() = CreateNode("b", {"a", "d"});
  *graph.add_node() = CreateNode("c", {"b", "d", "f"});
  *graph.add_node() = CreateNode("d", {"e"});
  *graph.add_node() = CreateNode("e", {"d"});
  *graph.add_node() = CreateNode("f", {"e", "g"});
  *graph.add_node() = CreateNode("g", {"f", "h"});
  *graph.add_node() = CreateNode("h", {"h"});

  std::vector<const NodeDef*> nodes;
  std::unordered_map<string, const NodeDef*> name_to_node;
  for (const auto& n : graph.node()) {
    nodes.push_back(&n);
    name_to_node[n.name()] = &n;
  }

  int num_components;
  std::unordered_map<const NodeDef*, int> components;
  StronglyConnectedComponents(graph, &components, &num_components);

  EXPECT_EQ(num_components, 4);
  for (const auto& pair :
       {std::make_pair("a", "b"), std::make_pair("a", "c"),
        std::make_pair("d", "e"), std::make_pair("f", "g")}) {
    EXPECT_EQ(components[name_to_node[pair.first]],
              components[name_to_node[pair.second]]);
  }

  for (const auto& pair :
       {std::make_pair("a", "d"), std::make_pair("a", "f"),
        std::make_pair("a", "h"), std::make_pair("d", "f"),
        std::make_pair("d", "h"), std::make_pair("f", "h")}) {
    EXPECT_NE(components[name_to_node[pair.first]],
              components[name_to_node[pair.second]]);
  }
}

TEST_F(SCCTest, TensorFlowLoop) {
  // Test graph produced in python using:
  /*
     with tf.Graph().as_default():
       i = tf.constant(0)
       c = lambda i: tf.less(i, 10)
       b = lambda i: tf.add(i, 1)
       r = tf.while_loop(c, b, [i])
       with open('/tmp/graph.txt', 'w') as f:
         f.write(str(tf.get_default_graph().as_graph_def()))
  */
  const string gdef_ascii = R"EOF(
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "while/Enter"
  op: "Enter"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "frame_name"
    value {
      s: "while/while/"
    }
  }
  attr {
    key: "is_constant"
    value {
      b: false
    }
  }
  attr {
    key: "parallel_iterations"
    value {
      i: 10
    }
  }
}
node {
  name: "while/Merge"
  op: "Merge"
  input: "while/Enter"
  input: "while/NextIteration"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Less/y"
  op: "Const"
  input: "^while/Merge"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "while/Less"
  op: "Less"
  input: "while/Merge"
  input: "while/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/LoopCond"
  op: "LoopCond"
  input: "while/Less"
}
node {
  name: "while/Switch"
  op: "Switch"
  input: "while/Merge"
  input: "while/LoopCond"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@while/Merge"
      }
    }
  }
}
node {
  name: "while/Identity"
  op: "Identity"
  input: "while/Switch:1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Add/y"
  op: "Const"
  input: "^while/Identity"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "while/Add"
  op: "Add"
  input: "while/Identity"
  input: "while/Add/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/NextIteration"
  op: "NextIteration"
  input: "while/Add"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Exit"
  op: "Exit"
  input: "while/Switch"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
versions {
  producer: 11
}
  )EOF";

  GrapplerItem item;
  CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &item.graph));

  std::unordered_map<const NodeDef*, int> components;
  int num_components;
  StronglyConnectedComponents(item.graph, &components, &num_components);

  EXPECT_EQ(num_components, 2);
  for (const auto& node : item.graph.node()) {
    if (node.name() == "Const" || node.name() == "while/Enter" ||
        node.name() == "while/Exit") {
      // These nodes are not part of the loop, they should be assigned the id
      // -1.
      EXPECT_EQ(-1, components[&node]);
    } else {
      // These nodes are part of the loop, they should be assigned a positive
      // id.
      EXPECT_LE(0, components[&node]);
    }
  }
}

}  // namespace grappler
}  // namespace tensorflow
