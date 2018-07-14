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

#ifdef TENSORFLOW_USE_ROCM

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "convert_graph.h"


namespace tensorflow {
class ROCmPass : public GraphOptimizationPass {
  public:
    ROCmPass() {}
    
    // Standard interface to run pass
    Status Run(const GraphOptimizationPassOptions& options);
};

#if 0
const OptimizationPassRegistry::Grouping kROCmPassGroup =
    OptimizationPassRegistry::POST_PARTITIONING;
REGISTER_OPTIMIZATION(kROCmPassGroup, 1, ROCmPass);
#else    
const OptimizationPassRegistry::Grouping kROCmPassGroup =
    OptimizationPassRegistry::POST_REWRITE_FOR_EXEC;
REGISTER_OPTIMIZATION(kROCmPassGroup, 0, ROCmPass);
#endif
    
Status ROCmPass::Run(
  const GraphOptimizationPassOptions& options) {
    if (options.graph == nullptr && options.partition_graphs == nullptr) {
        return Status::OK();
    }
    const char* enable_migraph = getenv("TF_ENABLE_MIGRAPH");
    if (enable_migraph != nullptr) {
        int env_val = atoi(enable_migraph);
        if (env_val == 0)
            return Status::OK();
    }
    
    auto convertGraph = [&](std::unique_ptr<Graph>* g) {
        // Get the ownership of a graph
        std::unique_ptr<Graph>* ng = std::move(g);
        tensorflow::rtglib::convert::ConvertGraphToRTG(ng, options.inputs);;
        // Return the ownership of a graph back
        g->reset(ng->release());
    };

    if (kROCmPassGroup != OptimizationPassRegistry::POST_PARTITIONING) {
        // For any pre-partitioning phase, a graph is stored in options.graph.
        convertGraph(options.graph);
    } else {
        // For post partitioning phase, graphs are stored in
        // options.partition_graphs.
        for (auto& pg : *options.partition_graphs) {
            convertGraph(&pg.second);
        }
    }
    
    return Status::OK();
}
                         
} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
