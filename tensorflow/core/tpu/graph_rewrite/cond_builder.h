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

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_COND_BUILDER_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_COND_BUILDER_H_

#include <string>

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Conditional builder.
// Convenience builder to make it easy to construct a conditional. E.g.,
//   Node* pred = ...;
//   CondBuilder cb("cond", g);
//   auto switch_var = cb.AddInput("var", DT_RESOURCE);
//   g->AddEdge(pred, 0, cb.pred(), 0);
// Will create the nodes of a conditional that takes as input a resource
// variable ("var") as input and that switches on pred.
//
// This currently only handles the case needed by distributed_tpu_rewrite_pass
// and is not completely general.
class CondBuilder {
 public:
  enum Branch { kElseBranch = 0, kThenBranch = 1 };

  CondBuilder(std::string name, std::string device, const NodeDebugInfo& debug,
              Graph* graph);

  // Returns node corresponding to the predicate input.
  Node* pred();

  // Returns node corresponding to switch_f branch of predicate switch.
  Node* switch_f();

  // Returns node corresponding to switch_t branch of predicate switch.
  Node* switch_t();

  // Returns node corresponding to control successor.
  Node* control_successor();

  // Returns the Switch node to feed a value of the given type into the
  // conditional.
  Status AddInput(const std::string& input_name, const DataType& type,
                  const std::string& device, const NodeDebugInfo& debug,
                  Node** input);

 private:
  Node* control_successor_;
  Node* switch_f_;
  Node* switch_t_;
  Node* pred_;
  Graph* const graph_;
  const std::string name_;
  const std::string device_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_COND_BUILDER_H_
