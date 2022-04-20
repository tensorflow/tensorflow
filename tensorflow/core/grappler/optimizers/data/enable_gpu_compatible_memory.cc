/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/enable_gpu_compatible_memory.h"

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kPrefetchDataset[] = "PrefetchDataset";
constexpr char UseGpuCompatAllocatorAttr[] = "use_gpu_compat_allocator";

bool HasUseGpuCompatAllocatorAttr(const NodeDef& node) {
    return node.attr().contains(UseGpuCompatAllocatorAttr);
}
}  // namespace

Status EnableGPUCompatibleMemory::OptimizeAndCollectStats(Cluster* cluster,
                                               const GrapplerItem& item,
                                               GraphDef* output,
                                               OptimizationStats* stats) {

  *output = item.graph;
  MutableGraphView graph(output);

  for (const NodeDef& node : item.graph.node()) {
    // find the prefetch op
    if (node.op() != kPrefetchDataset) {
      continue;
    }
    const NodeDef& prefetch_node = node;
    NodeDef* node_prior = graph_utils::GetInputNode(prefetch_node, graph);

    if (node_prior == nullptr) {
      VLOG(2) << "No op was found prior to the prefetch op!";
      return Status::OK();      
    }

    if (!HasUseGpuCompatAllocatorAttr(*node_prior))
    {
      VLOG(2) << "The `" << node_prior->op() << "` op does not have the "
        << UseGpuCompatAllocatorAttr << " attribute, hence the "
        << "OptimizeAndCollectStats is skipped";
      return Status::OK();
    }
    (*node_prior->mutable_attr())[UseGpuCompatAllocatorAttr].set_b(true);
    return Status::OK();  
  }
  return Status::OK();  
}


REGISTER_GRAPH_OPTIMIZER_AS(EnableGPUCompatibleMemory, "enable_gpu_compatible_memory");


}  // namespace grappler
}  // namespace tensorflow
