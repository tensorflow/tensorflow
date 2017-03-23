/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_GRAPH_REWRITER_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_GRAPH_REWRITER_H_

#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

//
class GraphRewriter {
 public:
  GraphRewriter(const GrapplerItem& item);

  void ForwardPreservedInputs(
      const NodeDef& original_node,
      const std::unordered_set<const NodeDef*>& nodes_to_delete,
      NodeDef* new_node);

 private:
  std::unordered_map<string, const NodeDef*> nodes_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_GRAPH_REWRITER_H_
