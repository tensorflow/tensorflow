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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SUBGRAPH_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SUBGRAPH_H_

#include <initializer_list>
#include <set>
#include <unordered_set>

#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"
#include "tensorflow/core/grappler/graph_analyzer/map_tools.h"
#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

// The description of a single subgraph for processing.
class Subgraph {
 public:
  // Identity of a single subgraph as a set of nodes.
  class Identity : public gtl::FlatSet<const GenNode*> {
   public:
    using InitializerList = std::initializer_list<GenNode*>;

    Identity() = default;
    Identity(InitializerList init);
    bool operator<(const Identity& other) const;
    bool operator==(const Identity& other) const;

    // Compute the hash.
    size_t Hash() const;
  };

  explicit Subgraph(Identity id) : id_(std::move(id)), hash_(id_.Hash()) {}

  // Construct by extending the parent identity with an extra node.
  Subgraph(const Identity& parent_id, GenNode* add_node);

  Subgraph() = delete;
  Subgraph(const Subgraph& other) = delete;
  void operator=(const Subgraph& other) = delete;

  // Order for building sets of subgraphs.
  bool operator<(const Subgraph& other) const { return this->id_ < other.id_; }
  // Support for hashed sets.
  bool operator==(const Subgraph& other) const {
    return this->id_ == other.id_;
  }
  size_t Hash() const { return hash_; }

  // Dump the subgraph information to a string.
  string Dump();

  // Extract this subgraph into a separate graph representation for signature
  // building, that includes only the links between the nodes in the subgraph
  // and drops all the external links. The result map should be clear before the
  // call.
  void ExtractForSignature(SigNodeMap* result);

  const Identity& id() const { return id_; }
  bool specific() const { return specific_; }
  void SetSpecific(bool value) { specific_ = value; }
  int32_t collation_count() const { return collation_count_; }
  void AddCollation(int32_t n = 1) { collation_count_ += n; }
  void ResetCollation() { collation_count_ = 1; }
  void MergeCollation(const Subgraph& other) {
    collation_count_ += other.collation_count_;
  }

 private:
  // Identity also serves as the list of nodes. It never changes throughout the
  // life of subgraph.
  Identity id_;
  size_t hash_;  // Cached from the identity.
  // Whether the dump should include the specific names of the nodes. The
  // non-specific (i.e. generic) subgraphs represent a collation of multiple
  // subgraphs.
  bool specific_ = true;
  // How many collated subgraphs are represented by this subgraph.
  int32_t collation_count_ = 1;
};

// Iteration of all links in a subgraph. This is more like Java iterators than
// the normal C++ iterators. It's simpler this way and there seems to be no
// major reason to make it a proper C++ iterator.
class SubgraphIterator {
 public:
  // Obviously an iterator is valid only until the original object
  // gets destroyed.
  explicit SubgraphIterator(const Subgraph::Identity* id);
  explicit SubgraphIterator(const Subgraph* sg) : SubgraphIterator(&sg->id()) {}

  // Check whether the built-in iterator is at the end.
  bool AtEnd() const { return id_it_ == id_->end(); }

  // Get the neighbor at the current iterator.
  // MUST NOT be called when AtEnd();
  const GenNode::LinkTarget& GetNeighbor() const {
    return link_map_it_->second[link_idx_];
  }

  // Get the node at the current iterator.
  // MUST NOT be called when AtEnd();
  const GenNode* GetNode() const { return *id_it_; }

  // Get the port leading to the neighbor at the current iterator.
  // MUST NOT be called when AtEnd();
  GenNode::Port GetPort() const { return link_map_it_->first; }

  // Increases the iterator.
  // Returns true if NOT AtEnd() after increasing the iterator.
  // Safe to call if already AtEnd().
  bool Next();

  // If there are more links at the same port, increases the iterator and
  // returns true. Otherwise leaves the iterator unchanged and returns false.
  bool NextIfSamePort();

  // Increases the iterator directly to the last position on the current port
  // (or if already there then doesn't increase). Equivalent to calling
  // NextIfSamePort() while it returns true, but faster.
  // Safe to call if already AtEnd().
  void SkipPort();

  // Increases the iterator directly to the last position on the current node.
  // Safe to call if already AtEnd().
  void SkipNode();

  // Returns true if the iterators are exactly the same.
  bool operator==(const SubgraphIterator& other) const;
  bool operator!=(const SubgraphIterator& other) const {
    return !(*this == other);
  }

 private:
  // After link_idx_ has been increased, make sure that it points to the
  // next valid element (or end) by increasing the higher levels of iteration if
  // needed.
  // Returns true if NOT AtEnd() after increasing the iterator.
  // NOT safe to call if already AtEnd().
  bool PropagateNext();

  // Identity of the subgraph being iterated over.
  const Subgraph::Identity* id_;

  // The current position, allowing to iterate through the links (see the
  // reasoning for it in the public section).
  //
  // (1) Iterator of the nodes in the subgraph.
  Subgraph::Identity::const_iterator id_it_;
  // (2) Iterator in the link map of the node.
  GenNode::LinkMap::const_iterator link_map_it_;
  // (3) Index in the vector of the links.
  int32_t link_idx_;
};

// A convenient way to store subgraphs: in a set of unique_ptrs. This way the
// addresses of subgraph objects will stay stable, and the objects themselves
// won't be copied.
class SubgraphPtrSet
    : public std::unordered_set<std::unique_ptr<Subgraph>,
                                HashAtPtr<std::unique_ptr<Subgraph>>,
                                EqAtPtr<std::unique_ptr<Subgraph>>> {
 public:
  // Attempts to extend the set by adding a new subgraph that gets created by
  // adding one node to the parent subgraph. If such a subgraph already exists,
  // returns nullptr, otherwise returns the pointer to the new subgraph.
  Subgraph* ExtendParent(const Subgraph::Identity& parent_id, GenNode* node);
};

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_SUBGRAPH_H_
