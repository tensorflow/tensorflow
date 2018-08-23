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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_PARALLEL_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_PARALLEL_H_

#include "tensorflow/core/framework/variable.pb.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

// Automatically parallelize a graph by splitting in the batch dimension.
class AutoParallel : public GraphOptimizer {
 public:
  AutoParallel(int num_replicas) : num_replicas_(num_replicas) {
    CHECK(num_replicas_ >= 2);
  }
  ~AutoParallel() override {}

  string name() const override { return "autoparallel"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;

 private:
  GraphDef graph_;
  std::map<string, NodeDef*> all_nodes_;
  std::set<std::pair<std::string, std::string>> gradients_;
  std::set<string> replica_nodes_;
  std::set<string> shared_nodes_;
  const GrapplerItem* item_;
  int num_replicas_;
  int num_gpus_;
  Status Initialize(const GrapplerItem& item);
  NodeDef* AddNodeNumReplicasConst(bool is_float, GraphDef* graph);
  NodeDef* AddNodeMaxLongConst(const string& name, GraphDef* graph);
  NodeDef* AddNodeDiv(const string& name, const string& input_a,
                      const string& input_b, GraphDef* graph);
  NodeDef* AddNodeAdd(const string& name, const std::set<string>& inps,
                      GraphDef* graph);
  NodeDef* AddNodeSparseAccumulator(const string& name, GraphDef* graph);
  NodeDef* AddNodeCast(const string& name, const string& input,
                       const DataType& src_dtype, const DataType& dst_dtype,
                       GraphDef* graph);
  NodeDef* AddNodeSparseAccumApply(const string& name,
                                   const string& accumulator,
                                   const string& max_long,
                                   const string& indices, const string& values,
                                   GraphDef* graph);
  NodeDef* AddNodeSparseAccumTakeGrad(const string& name,
                                      const string& accumulator,
                                      const string& num_replicas,
                                      const string& control, GraphDef* graph);
  NodeDef* AddNodeControl(const string& name, const std::set<string>& deps,
                          GraphDef* graph);
  void AddDenseAggregatedGrad(GraphDef* graph, NodeDef* num_replicas,
                              const std::string& grad_name,
                              std::string* new_grad_name);
  void AddSparseAggregatedGrad(GraphDef* graph, NodeDef* num_replicas,
                               const std::string& indices_name,
                               const std::string& values_name,
                               std::string* new_indices_name,
                               std::string* new_grad_name);
  void UpdateConsumers(
      const std::vector<std::pair<NodeDef*, int>>& grad_consumers,
      const std::string& new_grad_name);
  bool NotSharedNode(const string& name);
  void AddSharedNodes(GraphDef* graph);
  void AddOneReplica(GraphDef* graph, int number);
  void AddGradientAggregation(GraphDef* graph);
  void BuildGraph(GraphDef* graph);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_PARALLEL_H_
