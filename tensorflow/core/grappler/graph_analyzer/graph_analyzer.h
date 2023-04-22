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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_GRAPH_ANALYZER_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_GRAPH_ANALYZER_H_

#include <deque>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/graph_analyzer/map_tools.h"
#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"
#include "tensorflow/core/grappler/graph_analyzer/subgraph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

namespace test {
class GraphAnalyzerTest;
}  // end namespace test

// Finds all the subgraphs of a given size and groups them by equivalence.
class GraphAnalyzer {
 public:
  // Makes a copy of the graph.
  GraphAnalyzer(const GraphDef& graph, int subgraph_size);

  virtual ~GraphAnalyzer();

  // Performs the analysis and collects the subgraphs.
  Status Run();

  // Returns the subgraphs found in Run() printed to text.
  std::vector<string> DumpSubgraphs();

  // Prints the subgraphs found in Run() to stdout.
  Status OutputSubgraphs();

  // TODO(babkin): add a way to extract the subgraphs as direct data
  // structures and as protobufs, and to write protobufs to a RecordIO.

 private:
  GraphAnalyzer() = delete;
  GraphAnalyzer(const GraphAnalyzer&) = delete;
  void operator=(const GraphAnalyzer&) = delete;

  friend class tensorflow::grappler::graph_analyzer::test::GraphAnalyzerTest;

  // Builds the map of nodes from the original graph definition.
  Status BuildMap();

  // Using nodes_, finds all the subgraphs of size subgraph_size_ and places
  // them into result_.
  void FindSubgraphs();

  // Deletes from result_ the unacceptable subgraphs. Those include the
  // subgraphs where not all the inputs at a multi-input port are included (this
  // could happen if some of these inputs were reached and included through
  // different paths).
  void DropInvalidSubgraphs();

  // Deletes from result_ duplicate entries of equivalent topology.
  Status CollateResult();

  // Returns the raw subgraphs found in FindSubgraphs() printed to text.
  std::vector<string> DumpRawSubgraphs();

  // Finds and adds appropriately to either partial_ or result_ all the
  // subgraphs that can be created by extending the parent subgraph by one node.
  // Ignores the duplicates.
  void ExtendSubgraph(Subgraph* parent);

  // Extends the parent subgraph by adding another node (if it wasn't already
  // added) and all its non-control inputs in the link map range at once.
  // If the subgraph would grow over subgraph_size_, it gets ignored.
  void ExtendSubgraphAllOrNone(Subgraph* parent, const GenNode* node);
  // Same but adds one specific inbound port (even control) all-or-none.
  void ExtendSubgraphPortAllOrNone(Subgraph* parent, const GenNode* node,
                                   GenNode::Port port);
  // The common final step called by ExtendSubgraph*AllOrNone() methods.
  void AddExtendedSubgraph(Subgraph* parent, const Subgraph::Identity& id);

  // Returns true if this subgraph has any multi-inputs that aren't all-in or
  // all-out.
  bool HasInvalidMultiInputs(Subgraph* sg);

  // Graph to run the analysis on.
  GraphDef graph_;
  int subgraph_size_;

  // The enriched graph of parsed nodes and connections.
  GenNodeMap nodes_;
  // The resulting set of subgraphs.
  SubgraphPtrSet result_;
  // The subgraphs of partial size, stored while finding the result.
  SubgraphPtrSet partial_;
  // The subgraphs of partial size (stored in partial_) that are still waiting
  // to be extended.
  //
  // TODO(babkin): This is rather simple-minded, each subgraph is examined from
  // scratch, which means that all its internal links get iterated too. But it's
  // OK for the small subgraphs. This can be improved by keeping not just
  // subgraphs but iterators on the list, each of them having the list not-yet
  // examined nodes (and the link position of the next link to be examined for
  // the first node). This would add extra constant overhead, so the break-even
  // subgraph size is not clear yet.
  std::deque<Subgraph*> todo_;

  // The collation map by signature is designed to allow the removal of entries
  // and moving of the signature references from the keys of this map to the
  // outside world. Must be careful at inserting and removal: make sure that
  // when a new entry is inserted, its signature reference gets populated with
  // the same data as the key of the map, and that if a reference is moved out,
  // the map entry gets removed before that reference gets destroyed.
  struct CollationEntry {
    std::shared_ptr<Signature> sig;
    size_t count = 0;
  };
  using CollationMap =
      std::unordered_map<Signature*, CollationEntry, HashAtPtr<Signature*>,
                         EqAtPtr<Signature*> >;
  CollationMap collation_map_;

  // The entries are owned by collation_map_, so must be removed from
  // ordered_collation_ before removing them from collation_map_.
  struct ReverseLessByCount {
    bool operator()(CollationEntry* left, CollationEntry* right) const {
      return left->count > right->count;  // Reverse order.
    }
  };
  using CollationOrderByCount =
      std::multiset<CollationEntry*, ReverseLessByCount>;
  CollationOrderByCount ordered_collation_;
};

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_GRAPH_ANALYZER_H_
