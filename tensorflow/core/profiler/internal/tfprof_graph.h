/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

// Build a graph structure based on op inputs/outputs. The graph is a directed
// acyclic graph pointing *from outputs to inputs*.

#ifndef THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_GRAPH_H_
#define THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_GRAPH_H_

#include <deque>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/internal/tfprof_node.h"
#include "tensorflow/core/profiler/internal/tfprof_show.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_options.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {

// Organize tensorflow ops in a graph structure, pointing from output ops
// to input ops.
class TFGraph : public TFShow {
 public:
  explicit TFGraph(checkpoint::CheckpointReader* ckpt_reader)
      : TFShow(ckpt_reader), root_(nullptr) {}
  ~TFGraph() override {}

  void AddNode(TFGraphNode* node) override;

  void Build() override;

 private:
  const ShowNode* ShowInternal(const Options& opts,
                               Timeline* timeline) override;

  bool ShouldShowIfExtra(const ShowNode* node, const Options& opts,
                         int depth) const override {
    return true;
  }

  GraphNode* CreateParentNode(const string& name);

  std::vector<GraphNode*> SearchRoot(const std::vector<GraphNode*>& roots,
                                     const std::vector<string>& regexes,
                                     std::set<string>* visited);

  std::vector<GraphNode*> PrintGraph(const std::vector<GraphNode*> roots,
                                     const Options& opts, int depth,
                                     int last_ident, std::set<string>* visits);

  std::vector<GraphNode*> Account(const std::vector<GraphNode*>& roots,
                                  const Options& opts,
                                  std::set<string>* visits);

  void Format(const std::vector<GraphNode*> roots, string* display_str,
              GraphNodeProto* proto);

  MemoryTracker memory_tracker_;
  GraphNode* root_;
  std::vector<std::unique_ptr<NodeDef>> node_defs_;
  std::map<string, std::unique_ptr<TFGraphNode>> parent_nodes_;
  std::map<string, std::unique_ptr<GraphNode>> nodes_map_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_GRAPH_H_
