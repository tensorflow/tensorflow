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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_TFGRAPH_BUILDER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_TFGRAPH_BUILDER_H_

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace xla {
namespace hlo_graph_dumper {

// This constructs a tensorflow graph for HLO computations.
class HloTfGraphBuilder {
 public:
  HloTfGraphBuilder(const DebugOptions& debug_options = DebugOptions());

  // Adds a computation to the graph.
  Status AddComputation(const HloComputation& computation);

  const tensorflow::GraphDef& GetGraphDef() const;

 private:
  // Gets the node name of an instruction. The node name is hierarchical. For
  // example, if an instruction is fused, it will be put in a subgraph of the
  // fusion instruction.
  const string& GetNodeNameForInstruction(const HloInstruction* instruction);

  void SetNodeAttrs(const HloInstruction* instruction,
                    tensorflow::NodeDef* node_def) const;

  Status AddInstruction(const HloInstruction* instruction);

  DebugOptions debug_options_;
  tensorflow::GraphDef graph_def_;
  // This records instructions that have been visited.
  std::unordered_set<const HloInstruction*> visited_instructions_;
  // A cache that maps instruction to the node name.
  std::unordered_map<const HloInstruction*, string> instruction_to_node_name_;
};

}  // namespace hlo_graph_dumper
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_TFGRAPH_BUILDER_H_
