/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_GRAPH_VALIDATE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_GRAPH_VALIDATE_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {
namespace graph {

// Returns OK if 'graph_def' has the following properties:
//
// 1) Every NodeDef is valid with respect to its corresponding OpDef
//    as registered in 'op_registry'.
//
// REQUIRES: 'op_registry' is not nullptr.
Status ValidateGraphDef(const GraphDef& graph_def,
                        const OpRegistryInterface* op_registry);

}  // namespace graph
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_GRAPH_VALIDATE_H_
