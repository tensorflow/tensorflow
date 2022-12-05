/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/compiler/tf2tensorrt/convert/trt_layout_optimization_pass.h"

#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stacktrace.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {
namespace convert {

using absl::AsciiStrToUpper;
using absl::StrAppend;
using absl::StrCat;

TRTLayoutOptimizationPass::TRTLayoutOptimizationPass(const string& name)
    : name_(name),
      trt_logger_name_("DefaultLogger"),
      minimum_segment_size_(3),
      is_dynamic_op_(false),
      max_cached_batches_(1),
      max_workspace_size_bytes_(256LL << 20) {
  VLOG(1) << "Constructing " << name_;
}

Status TRTLayoutOptimizationPass::Optimize(grappler::Cluster* cluster,
                                           const grappler::GrapplerItem& item,
                                           GraphDef* optimized_graph) {
  GraphDef modified_graph_def = item.graph;

  // Construct a GrapplerItem using the modified graph_def and the input
  // grappler_item.
  grappler::GrapplerItem grappler_item =
      grappler_item.WithGraph(std::move(modified_graph_def));
  const GraphDef& graph_def = grappler_item.graph;

  // Convert graphdef to graph.
  FunctionLibraryDefinition flib(OpRegistry::Global(), graph_def.library());
  Graph graph(flib);
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(GraphConstructorOptions(), graph_def, &graph));

  // Algorithm steps:
  // 1. We iterate over the graph to find any Conv (or other layout sensitive
  // op)
  // 2. If found, we continue, else we return
  // 3. We iterate over the nodes and replace the layout-sensitive params
  // 3. We add Transpose before the inputs and after the outputs

  grappler::GraphProperties static_graph_properties(grappler_item);

  std::cout << "TRTLayoutOptimizationPass: reading nodes..." << std::endl;
  for (Node* node : graph.nodes()) {
    std::cout << node->name() << std::endl;
  }

  // TODO: assign output *optimized_graph =;
}

Status TRTLayoutOptimizationPass::Init(
    const RewriterConfig_CustomGraphOptimizer* config) {
  std::cout << "Do nothing for now" << std::endl;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
