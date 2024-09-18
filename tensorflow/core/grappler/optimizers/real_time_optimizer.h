#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_REAL_TIME_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_REAL_TIME_OPTIMIZER_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/memory_optimizer.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

class RealTimeOptimizer : public CustomGraphOptimizer {
 public:
  RealTimeOptimizer();
  explicit RealTimeOptimizer(const RewriterConfig::Toggle opt_level);
  ~RealTimeOptimizer() override {}

  string name() const override { return "real_time_optimizer"; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item, GraphDef* output) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item, const GraphDef& optimized_graph, double result);

  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config = nullptr) override;

  bool UsesFunctionLibrary() const override;

 private:
  Status Initialize(Cluster* cluster, const GrapplerItem& item);
  Status ApplyOptimizations(GraphDef* optimized_graph);
  Status OptimizeGraph(GraphDef* optimized_graph);

  Status AdjustMemoryAllocations(GraphDef* optimized_graph);
  Status PerformOperationFusion(GraphDef* optimized_graph);

  bool CanOptimizeNode(const NodeDef& node, const GraphProperties& properties);
  void OptimizeNode(NodeDef* node, NodeMap* node_map);
  void ModifyInputAttributes(NodeDef* input_node);

  RewriterConfig::Toggle opt_level_;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_REAL_TIME_OPTIMIZER_H_