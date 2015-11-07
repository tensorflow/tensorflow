#ifndef TENSORFLOW_GRAPH_DEFAULT_DEVICE_H_
#define TENSORFLOW_GRAPH_DEFAULT_DEVICE_H_

#include <string>

#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {
namespace graph {

// Sets the default device for all nodes in graph_def to "device",
// only if not already set.
inline void SetDefaultDevice(const string& device, GraphDef* graph_def) {
  for (int i = 0; i < graph_def->node_size(); ++i) {
    auto node = graph_def->mutable_node(i);
    if (node->device().empty()) {
      node->set_device(device);
    }
  }
}

}  // namespace graph
}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_DEFAULT_DEVICE_H_
