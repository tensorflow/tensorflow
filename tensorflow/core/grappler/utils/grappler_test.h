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

#ifndef TENSORFLOW_GRAPPLER_GRAPPLER_TEST_H_
#define TENSORFLOW_GRAPPLER_GRAPPLER_TEST_H_

#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class GrapplerTest : public ::testing::Test {
 protected:
  std::vector<Tensor> EvaluateNodes(const GraphDef& graph,
                                    const std::vector<string>& node_names);

  std::vector<Tensor> EvaluateFetchNodes(const GrapplerItem& item);

  void AddNode(const string& name, const string& op,
               const std::vector<string>& inputs, GraphDef* graph);

  void CompareGraphs(GraphDef want, GraphDef got);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_GRAPPLER_TEST_H_
