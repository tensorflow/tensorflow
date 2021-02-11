/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PYTREE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PYTREE_H_

// See https://jax.readthedocs.io/en/latest/pytrees.html for the documentation
// about pytree.

// Caution: this code uses exceptions. The exception use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace xla {

// Registry of custom node types.
class CustomNodeRegistry {
 public:
  struct Registration {
    // The Python type object, used to identify the type.
    pybind11::object type;
    // A function with signature: object -> (iterable, aux_data)
    pybind11::function to_iterable;
    // A function with signature: (aux_data, iterable) -> object
    pybind11::function from_iterable;
  };

  // Registers a new custom type. Objects of `type` will be treated as container
  // node types in PyTrees.
  static void Register(pybind11::object type, pybind11::function to_iterable,
                       pybind11::function from_iterable);

  // Finds the custom type registration for `type`. Returns nullptr if none
  // exists.
  static const Registration* Lookup(pybind11::handle type);

 private:
  static CustomNodeRegistry* Singleton();

  struct TypeHash {
    size_t operator()(const pybind11::object& t) const {
      return pybind11::hash(t);
    }
  };
  struct TypeEq {
    bool operator()(const pybind11::object& a,
                    const pybind11::object& b) const {
      return a.equal(b);
    }
  };
  absl::flat_hash_map<pybind11::object, std::unique_ptr<Registration>, TypeHash,
                      TypeEq>
      registrations_;
};

// A PyTreeDef describes the tree structure of a PyTree. A PyTree is a tree of
// Python values, where the interior nodes are tuples, lists, dictionaries, or
// user-defined containers, and the leaves are other objects.
class PyTreeDef {
 public:
  PyTreeDef() = default;

  // Flattens a Pytree into a list of leaves and a PyTreeDef.
  static std::pair<std::vector<pybind11::object>, std::unique_ptr<PyTreeDef>>
  Flatten(pybind11::handle x,
          absl::optional<pybind11::function> leaf_predicate = absl::nullopt);

  // Recursive helper used to implement Flatten().
  void FlattenInto(
      pybind11::handle handle, std::vector<pybind11::object>& leaves,
      absl::optional<pybind11::function> leaf_predicate = absl::nullopt);

  // Tests whether the given list is a flat list of leaves.
  static bool AllLeaves(const pybind11::iterable& x);

  // Flattens a Pytree up to this PyTreeDef. 'this' must be a tree prefix of
  // the tree-structure of 'x'. For example, if we flatten a value
  // [(1, (2, 3)), {"foo": 4}] with a treedef [(*, *), *], the result is the
  // list of leaves [1, (2, 3), {"foo": 4}].
  pybind11::list FlattenUpTo(pybind11::handle x) const;

  // Returns an unflattened PyTree given an iterable of leaves and a PyTreeDef.
  pybind11::object Unflatten(pybind11::iterable leaves) const;

  // Composes two PyTreeDefs, replacing the leaves of this tree with copies of
  // `inner`.
  std::unique_ptr<PyTreeDef> Compose(const PyTreeDef& inner) const;

  // Makes a Tuple PyTreeDef out of a vector of PyTreeDefs.
  static std::unique_ptr<PyTreeDef> Tuple(const std::vector<PyTreeDef>& defs);

  std::vector<std::unique_ptr<PyTreeDef>> Children() const;

  // Maps a function over a PyTree structure, applying f_leaf to each leaf, and
  // f_node to each container node.
  // TODO(phawkins): use flattening everywhere instead and delete this method.
  pybind11::object Walk(const pybind11::function& f_node,
                        pybind11::handle f_leaf,
                        pybind11::iterable leaves) const;

  // Given a tree of iterables with the same node/leaf structure as this PyTree,
  // build the corresponding PyTree.
  // TODO(phawkins): use flattening everywhere instead and delete this method.
  pybind11::object FromIterableTree(pybind11::handle xs) const;

  int num_leaves() const {
    if (traversal_.empty()) {
      return 0;
    }
    return traversal_.back().num_leaves;
  }

  int num_nodes() const { return traversal_.size(); }

  size_t Hash() const;

  bool operator==(const PyTreeDef& other) const;
  bool operator!=(const PyTreeDef& other) const { return !(*this == other); }

  std::string ToString() const;

 private:
  enum class Kind {
    kLeaf,        // An opaque leaf node
    kNone,        // None.
    kTuple,       // A tuple
    kNamedTuple,  // A collections.namedtuple
    kList,        // A list
    kDict,        // A dict
    kCustom,      // A custom type.
  };

  struct Node {
    Kind kind = Kind::kLeaf;

    // Arity for non-kLeaf types.
    int arity = 0;

    // Kind-specific auxiliary data. For a kNamedTuple, contains the tuple type
    // object. For a kDict, contains a sorted list of keys. For a kCustom type,
    // contains the auxiliary data returned by the `to_iterable` function.
    pybind11::object node_data;

    const CustomNodeRegistry::Registration* custom = nullptr;

    // Number of leaf nodes in the subtree rooted at this node.
    int num_leaves = 0;

    // Number of leaf and interior nodes in the subtree rooted at this node.
    int num_nodes = 0;
  };
  template <typename H>
  friend H AbslHashValue(H h, const Node& n);

  template <typename H>
  friend H AbslHashValue(H h, const PyTreeDef& t);

  // Helper that manufactures an instance of a node given its children.
  static pybind11::object MakeNode(const Node& node,
                                   absl::Span<pybind11::object> children);

  // Recursive helper used to implement FromIterableTree()
  pybind11::object FromIterableTreeHelper(
      pybind11::handle xs,
      std::vector<PyTreeDef::Node>::const_reverse_iterator* it) const;

  // Computes the node kind of a given Python object.
  static Kind GetKind(const pybind11::handle& obj,
                      CustomNodeRegistry::Registration const** custom);

  // Nodes, in a post-order traversal. We use an ordered traversal to minimize
  // allocations, and post-order corresponds to the order we need to rebuild the
  // tree structure.
  std::vector<Node> traversal_;
};

template <typename H>
H AbslHashValue(H h, const PyTreeDef::Node& n) {
  h = H::combine(std::move(h), n.kind, n.arity, n.custom);
  return h;
}

template <typename H>
H AbslHashValue(H h, const PyTreeDef& t) {
  return H::combine_contiguous(std::move(h), t.traversal_.data(),
                               t.traversal_.size());
}

void BuildPytreeSubmodule(pybind11::module& m);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PYTREE_H_
