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
#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SCOPED_ALLOCATOR_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SCOPED_ALLOCATOR_OPTIMIZER_H_

#include <atomic>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
class Graph;

namespace grappler {
class GraphProperties;
class NodeMap;
class ScopedAllocatorOptimizer;

// An Optimizer that introduces ScopedAllocators in order to reduce data
// movement and consolidate some kinds of Ops.
class ScopedAllocatorOptimizer : public GraphOptimizer {
 public:
  ScopedAllocatorOptimizer(RewriterConfig::Toggle opt_level,
                           const ScopedAllocatorOptions& opts);
  ~ScopedAllocatorOptimizer() override;

  string name() const override { return "scoped_allocator_optimizer"; }

  bool UsesFunctionLibrary() const override { return true; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override;

  // Map from an Op name to a vector of Nodes with that Op.
  typedef absl::flat_hash_map<string, std::vector<NodeDef*>> DevOpOccurrences;
  // Map from a device name to a DevOpOccurrences map.
  typedef absl::flat_hash_map<string, DevOpOccurrences> GraphOpOccurrences;
  typedef absl::flat_hash_set<string> OpNameSet;

  absl::Status ProcessGraphDef(GraphDef* graph,
                               const GraphProperties& graph_properties);

  // Populates *occs by grouping Nodes with common Ops, according to
  // their assigned devices.
  void FindOpOccurrences(GraphDef* graph, const OpNameSet& op_names,
                         GraphOpOccurrences* occs);

  // Returns a new, unused scope_id to be assigned to a ScopedAllocator that
  // will allocate num_fields (> 0) separate tensors.
  int NewScopedAllocatorId(int num_fields);

  // Returns a new, unused id to be assigned to an IdentityOp used in this graph
  // rewrite.
  absl::Status NewIdentityId(int* id);

  NodeMap* node_map() { return node_map_.get(); }

  const absl::flat_hash_set<string>& repeated_outputs() {
    return repeated_outputs_;
  }

  // Appends values to the attr value under name in node_def, if present.
  // If not present does an assignment.
  static void ExtendNodeAttr(absl::string_view name,
                             const std::vector<int32>& values,
                             NodeDef* node_def);

  // Class that knows how to do graph rewriting for a particular kind of Op in
  // order to take advantage of a ScopedAllocator.
  class Rewriter {
   public:
    virtual ~Rewriter() {}

    virtual absl::Status Rewrite(ScopedAllocatorOptimizer* paopti,
                                 int64_t invocation_count, GraphDef* graph,
                                 const string& op_name,
                                 const std::vector<NodeDef*>& nodes,
                                 bool* applied) = 0;

    void SetGraphProperties(const GraphProperties& graph_properties) {
      graph_properties_ = &graph_properties;
      CHECK(graph_properties_);
    }

   protected:
    const GraphProperties* graph_properties_;
  };

 private:
  Rewriter* GetRewriter(const string& op_name);

  absl::Status OrderNodeSet(std::vector<NodeDef*>* nodes) const;

  RewriterConfig::Toggle opt_level_;
  std::unordered_set<string> nodes_to_preserve_;
  OpNameSet op_name_set_;
  absl::flat_hash_map<string, Rewriter*> rewriters_;
  std::vector<Rewriter*> to_delete_;
  int next_sa_id_ = 1;
  int next_identity_id_ = 1;
  std::unique_ptr<NodeMap> node_map_;
  // Keeps track of outputs, i.e. a node and an output index, that are inputs to
  // more than one op groups that are candidates for scoped allocator
  // optimization.
  absl::flat_hash_set<string> repeated_outputs_;
};

}  // namespace grappler
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SCOPED_ALLOCATOR_OPTIMIZER_H_
