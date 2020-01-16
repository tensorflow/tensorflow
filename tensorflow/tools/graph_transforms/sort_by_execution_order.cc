/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// This is a thin wrapper with the standard TransformFunc interface to the
// underlying utility function. The only difference is that we don't use the
// input or output name arguments.
Status SortByExecutionOrderWithUnusedContext(
    const GraphDef& input_graph_def, const TransformFuncContext& unused_context,
    GraphDef* output_graph_def) {
  return SortByExecutionOrder(input_graph_def, output_graph_def);
}

REGISTER_GRAPH_TRANSFORM("sort_by_execution_order",
                         SortByExecutionOrderWithUnusedContext);

}  // namespace graph_transforms
}  // namespace tensorflow
