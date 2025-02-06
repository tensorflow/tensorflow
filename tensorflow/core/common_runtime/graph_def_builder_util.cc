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
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"

#include "tensorflow/core/common_runtime/graph_constructor.h"

namespace tensorflow {

absl::Status GraphDefBuilderToGraph(const GraphDefBuilder& builder,
                                    Graph* graph) {
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(builder.ToGraphDef(&graph_def));
  GraphConstructorOptions opts;
  return ConvertGraphDefToGraph(opts, std::move(graph_def), graph);
}

}  // namespace tensorflow
