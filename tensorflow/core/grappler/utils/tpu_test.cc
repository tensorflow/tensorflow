/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/tpu.h"

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class TpuTest : public ::testing::Test {};

TEST_F(TpuTest, NotTpuGraph) {
  // Test where no TPU op exists.
  {
    GraphDef tpu_graph;
    tpu_graph.add_node()->set_op("Add");
    FunctionDefLibrary* library = tpu_graph.mutable_library();
    FunctionDef* function_def = library->add_function();
    function_def->add_node_def()->set_op("Mul");
    EXPECT_FALSE(IsTPUGraphDef(tpu_graph));
  }
}

TEST_F(TpuTest, TpuMainGraph) {
  // Test where TPU op is in main graph.
  {
    GraphDef tpu_graph;
    tpu_graph.add_node()->set_op("TPUPartitionedCall");
    EXPECT_TRUE(IsTPUGraphDef(tpu_graph));
  }
}

TEST_F(TpuTest, TpuLibraryGraph) {
  // Test where the TPU Graph is not called directly from the main graph.
  {
    GraphDef tpu_graph;
    tpu_graph.add_node()->set_op("BatchFunction");
    FunctionDefLibrary* library = tpu_graph.mutable_library();
    FunctionDef* function_def = library->add_function();
    function_def->add_node_def()->set_op("TPUPartitionedCall");
    EXPECT_TRUE(IsTPUGraphDef(tpu_graph));
  }
}

}  // end namespace grappler
}  // end namespace tensorflow
