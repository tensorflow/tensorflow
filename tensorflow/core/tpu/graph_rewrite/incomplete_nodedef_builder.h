/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_INCOMPLETE_NODEDEF_BUILDER_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_INCOMPLETE_NODEDEF_BUILDER_H_

#include <string>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Convenience builder to build NodeDefs without specifying the inputs. This is
// similar to NodeDefBuilder except inputs are not specified.
// TODO(jpienaar): Clean up NodeDefBuilder and remove this class.
class IncompleteNodeDefBuilder {
 public:
  IncompleteNodeDefBuilder(const string& name, const string& op,
                           const NodeDebugInfo& debug);

  IncompleteNodeDefBuilder& AddAttr(const string& attr, const DataType& type);
  IncompleteNodeDefBuilder& AddAttr(const string& attr, int val);

  IncompleteNodeDefBuilder& Device(const string& device);

  Status Build(Graph* graph, Node** n);

  static IncompleteNodeDefBuilder Identity(const string& name,
                                           const DataType& type,
                                           const NodeDebugInfo& debug);
  static IncompleteNodeDefBuilder Merge(const string& name,
                                        const DataType& type,
                                        const NodeDebugInfo& debug, int n);
  static IncompleteNodeDefBuilder Switch(const string& name,
                                         const DataType& type,
                                         const NodeDebugInfo& debug);

 private:
  NodeDef nodedef_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_INCOMPLETE_NODEDEF_BUILDER_H_
