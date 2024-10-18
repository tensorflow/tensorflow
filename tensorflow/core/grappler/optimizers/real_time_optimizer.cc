// real_time_optimizer.cc

#include "tensorflow/core/grappler/optimizers/real_time_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {

RealTimeOptimizer::RealTimeOptimizer() {
  VLOG(1) << "Initializing RealTimeOptimizer.";
}

Status RealTimeOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                   GraphDef* optimized_graph) {
  VLOG(1) << "Running RealTimeOptimizer on graph: " << item.id;

  *optimized_graph = item.graph;

  GraphProperties properties(item);
  TF_RETURN_IF_ERROR(properties.InferStatically(false));

  NodeMap node_map(optimized_graph);

  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    NodeDef* node = optimized_graph->mutable_node(i);

    if (CanOptimizeNode(*node, properties)) {
      OptimizeNode(node, &node_map);
    }
  }

  VLOG(1) << "Completed running RealTimeOptimizer on graph: " << item.id;
  return OkStatus();
}

Status RealTimeOptimizer::Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  VLOG(1) << "Initializing RealTimeOptimizer with custom configuration.";
  if (config != nullptr) {
    // Use configuration settings from `config` if needed.
  }
  return OkStatus();
}

bool RealTimeOptimizer::UsesFunctionLibrary() const {
  return false;  
}

bool RealTimeOptimizer::CanOptimizeNode(const NodeDef& node,
                                        const GraphProperties& properties) {
  if (node.op() == "MatMul") {
    const AttrValue* transpose_a_attr = AttrSlice(node).Find("transpose_a");
    const AttrValue* transpose_b_attr = AttrSlice(node).Find("transpose_b");
    if (transpose_a_attr && transpose_b_attr &&
        !transpose_a_attr->b() && !transpose_b_attr->b()) {
      return true;
    }
  }
  return false;
}

void RealTimeOptimizer::OptimizeNode(NodeDef* node, NodeMap* node_map) {
  if (node->op() == "MatMul") {
    node->set_op("OptimizedMatMul");
    VLOG(1) << "Optimized MatMul node: " << node->name();

    for (int i = 0; i < node->input_size(); ++i) {
      const string& input = node->input(i);
      NodeDef* input_node = node_map->GetNode(input);
      if (input_node) {
        ModifyInputAttributes(input_node);
      }
    }
  }
}

void RealTimeOptimizer::ModifyInputAttributes(NodeDef* input_node) {
  AttrValue attr_value;
  attr_value.set_b(true);
  (*input_node->mutable_attr())["optimized"] = attr_value;
  VLOG(1) << "Modified input node attributes for: " << input_node->name();
}

REGISTER_GRAPH_OPTIMIZER_AS(RealTimeOptimizer, "real_time_optimizer");

}  // namespace grappler
}  // namespace tensorflow