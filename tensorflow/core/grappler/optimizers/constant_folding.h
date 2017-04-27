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

#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

const char kConstantFoldingConst[] = "ConstantFolding";

// Contant folding optimization for a graph.
class ConstantFolding : public GraphOptimizer {
 public:
  ConstantFolding() {}

  ~ConstantFolding() override {}

  string name() const override { return "constant folding"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;

 private:
  bool IsConst(const NodeDef& node) const;

  bool IsFoldable(const NodeDef& node) const;

  NodeDef CreateNodeDef(const string& name, const TensorValue& tensor);

  Status EvaluateNode(const NodeDef& node,
                      const gtl::InlinedVector<TensorValue, 4>& inputs,
                      gtl::InlinedVector<TensorValue, 4>* output);

  Status EvaluateOneFoldable(const NodeDef& node,
                             std::vector<NodeDef>* outputs);

  Status FoldNode(const NodeDef& node, GraphDef* output);

  Status FoldGraph(GraphDef* output);

  std::unique_ptr<DeviceBase> device_;
  GraphDef graph_;
  std::unique_ptr<NodeMap> node_map_;
  std::set<string> nodes_to_preserve_;
  std::set<string> ops_to_preserve_ = {"Save",    "SaveV2",    "SaveSlices",
                                       "Restore", "RestoreV2", "RestoreSlice"};
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_
