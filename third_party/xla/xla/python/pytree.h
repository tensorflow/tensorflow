/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PYTREE_H_
#define XLA_PYTHON_PYTREE_H_

// See https://jax.readthedocs.io/en/latest/pytrees.html for the documentation
// about pytree.

#include <Python.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// placeholder for index annotation headers
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/pytree.pb.h"

namespace xla {

enum class PyTreeKind {
  kLeaf,        // An opaque leaf node
  kNone,        // None.
  kTuple,       // A tuple
  kNamedTuple,  // A collections.namedtuple
  kList,        // A list
  kDict,        // A dict
  kCustom,      // A custom type.
  kDataclass,   // A dataclass.
};

// Registry of custom node types.
class PyTreeRegistry {
 public:
  PyTreeRegistry(bool enable_none, bool enable_tuple, bool enable_namedtuple,
                 bool enable_list, bool enable_dict);

  PyTreeRegistry(const PyTreeRegistry&) = delete;
  PyTreeRegistry(PyTreeRegistry&&) = delete;
  PyTreeRegistry& operator=(const PyTreeRegistry&) = delete;
  PyTreeRegistry& operator=(PyTreeRegistry&&) = delete;

  struct Registration {
    PyTreeKind kind;

    // The following values are populated for custom types.
    // The Python type object, used to identify the type.
    nanobind::object type;
    // A function with signature: object -> (iterable, aux_data)
    nanobind::callable to_iterable;
    // A function with signature: (aux_data, iterable) -> object
    nanobind::callable from_iterable;
    // A function with signature: (aux_data, iterable(keypath, leaf)) -> object
    std::optional<nanobind::callable> to_iterable_with_keys;

    // Helper that calls to_iterable and validates that it returns a pair
    // of an iterable and an aux_data object
    std::pair<nanobind::iterable, nanobind::object> ToIterable(
        nanobind::handle o) const;
    // Helper that calls to_iterable_with_keys and validates that it returns a
    // pair of an iterable of key-leaf pairs and an aux_data object. If
    // to_iterable_with_keys is not available, return a dummy key for each leaf,
    // similar to the current jax.tree_util.FlattenedIndexKey.
    std::pair<std::vector<std::pair<nanobind::object, nanobind::object>>,
              nanobind::object>
    ToIterableWithKeys(nanobind::handle o) const;

    // For dataclasses.
    std::vector<nanobind::str> data_fields;
    std::vector<nanobind::str> meta_fields;

    int tp_traverse(visitproc visit, void* arg);
  };

  // Registers a new custom type. Objects of `type` will be treated as container
  // node types in PyTrees.
  void Register(
      nanobind::object type, nanobind::callable to_iterable,
      nanobind::callable from_iterable,
      std::optional<nanobind::callable> to_iterable_with_keys = std::nullopt);
  // Same, but for dataclasses.
  void RegisterDataclass(nanobind::object type,
                         std::vector<nanobind::str> data_fields,
                         std::vector<nanobind::str> meta_fields);

  // Finds the custom type registration for `type`. Returns nullptr if none
  // exists.
  const Registration* Lookup(nanobind::handle type) const;

  PyTreeKind KindOfObject(nanobind::handle obj,
                          PyTreeRegistry::Registration const** custom) const;

  // Flattens a pytree one level, returning either a tuple of the leaves and
  // the node data, or None, if the entry is a leaf.
  nanobind::object FlattenOneLevel(nanobind::handle x) const;

  static PyType_Slot slots_[];

 private:
  struct TypeHash {
    using is_transparent = void;
    size_t operator()(const nanobind::object& t) const {
      return absl::HashOf(t.ptr());
    }
    size_t operator()(const nanobind::handle& t) const {
      return absl::HashOf(t.ptr());
    }
  };
  struct TypeEq {
    using is_transparent = void;
    bool operator()(const nanobind::object& a,
                    const nanobind::object& b) const {
      return a.ptr() == b.ptr();
    }
    bool operator()(const nanobind::object& a,
                    const nanobind::handle& b) const {
      return a.ptr() == b.ptr();
    }
  };
  absl::flat_hash_map<nanobind::object, std::unique_ptr<Registration>, TypeHash,
                      TypeEq>
      registrations_;
  bool enable_namedtuple_;

  static int tp_traverse(PyObject* self, visitproc visit, void* arg);
  static int tp_clear(PyObject* self);
};

class SequenceKey {
 public:
  explicit SequenceKey(int idx) : idx_(idx) {};
  std::string ToReprString() const;
  std::string ToString() const;
  bool Equals(const nanobind::object& other);
  int idx() const { return idx_; }
  static nanobind::tuple MatchArgs(nanobind::handle unused);

 private:
  int idx_;
};

class DictKey {
 public:
  explicit DictKey(nanobind::object key) : key_(key) {};
  std::string ToReprString() const;
  std::string ToString() const;
  bool Equals(const nanobind::object& other);
  nanobind::object key() const { return key_; }
  static nanobind::tuple MatchArgs(nanobind::handle unused);
  static PyType_Slot slots_[];

 private:
  nanobind::object key_;
  static int tp_traverse(PyObject* self, visitproc visit, void* arg);
  static int tp_clear(PyObject* self);
};

class GetAttrKey {
 public:
  explicit GetAttrKey(nanobind::str name) : name_(name) {};
  std::string ToReprString() const;
  std::string ToString() const;
  bool Equals(const nanobind::object& other);
  nanobind::str name() const { return name_; }
  static nanobind::tuple MatchArgs(nanobind::handle unused);

 private:
  nanobind::str name_;
};

class FlattenedIndexKey {
 public:
  explicit FlattenedIndexKey(int key) : key_(key) {};
  std::string ToReprString() const;
  std::string ToString() const;
  bool Equals(const nanobind::object& other);
  int key() const { return key_; }
  static nanobind::tuple MatchArgs(nanobind::handle unused);

 private:
  int key_;
};

// A PyTreeDef describes the tree structure of a PyTree. A PyTree is a tree of
// Python values, where the interior nodes are tuples, lists, dictionaries, or
// user-defined containers, and the leaves are other objects.
class PyTreeDef {
 public:
  // Unowned registry: the registry must remain live at least as long as the
  // PyTreeDef. It is the caller's responsibility to enforce this.
  explicit PyTreeDef(PyTreeRegistry* registry) : registry_(registry) {}

  explicit PyTreeDef(nb_class_ptr<PyTreeRegistry> registry)
      : registry_(registry.get()), registry_ref_(std::move(registry)) {}

  // Flattens a Pytree into a list of leaves and a PyTreeDef.
  // Returns references to the flattened objects, which might be temporary
  // objects in the case of custom pytype handlers.
  static std::pair<std::vector<nanobind::object>, nb_class_ptr<PyTreeDef>>
  Flatten(nanobind::handle x, nb_class_ptr<PyTreeRegistry> registry,
          std::optional<nanobind::callable> leaf_predicate = std::nullopt);

  // Flattens a Pytree into a list of `leaves` and a PyTreeDef (this).
  // `leaves` owns references to the flattened objects, which might be
  // temporary objects in the case of custom pytype handlers.
  void Flatten(nanobind::handle handle, std::vector<nanobind::object>& leaves,
               std::optional<nanobind::callable> leaf_predicate = std::nullopt);
  void Flatten(nanobind::handle handle,
               absl::InlinedVector<nanobind::object, 2>& leaves,
               std::optional<nanobind::callable> leaf_predicate = std::nullopt);
  void Flatten(nanobind::handle handle, nanobind::list& leaves,
               std::optional<nanobind::callable> leaf_predicate = std::nullopt);

  void FlattenWithPath(
      nanobind::handle handle, nanobind::list& leaves,
      std::optional<nanobind::callable> leaf_predicate = std::nullopt);

  // Tests whether the given list is a flat list of leaves.
  static bool AllLeaves(PyTreeRegistry* registry, const nanobind::iterable& x);

  // Flattens a Pytree up to this PyTreeDef. 'this' must be a tree prefix of
  // the tree-structure of 'x'. For example, if we flatten a value
  // [(1, (2, 3)), {"foo": 4}] with a treedef [(*, *), *], the result is the
  // list of leaves [1, (2, 3), {"foo": 4}].
  nanobind::list FlattenUpTo(nanobind::handle x) const;

  // Returns an unflattened PyTree given an iterable of leaves and a PyTreeDef.
  nanobind::object Unflatten(nanobind::iterable leaves) const;
  nanobind::object Unflatten(absl::Span<const nanobind::object> leaves) const;

  // Composes two PyTreeDefs, replacing the leaves of this tree with copies of
  // `inner`. The returned PyTreeDef holds a reference to its registry.
  nb_class_ptr<PyTreeDef> Compose(const PyTreeDef& inner) const;

  // Makes a Tuple PyTreeDef out of a vector of PyTreeDefs.
  static nb_class_ptr<PyTreeDef> Tuple(nb_class_ptr<PyTreeRegistry> registry,
                                       nanobind::list defs);

  // The returned PyTreeDefs hold a reference to the registry.
  std::vector<nb_class_ptr<PyTreeDef>> Children() const;

  // Maps a function over a PyTree structure, applying f_leaf to each leaf, and
  // f_node(node, node_data) to each container node.
  nanobind::object Walk(const nanobind::callable& f_node,
                        nanobind::handle f_leaf,
                        nanobind::iterable leaves) const;

  // Given a tree of iterables with the same node/leaf structure as this PyTree,
  // build the corresponding PyTree.
  // TODO(phawkins): use flattening everywhere instead and delete this method.
  nanobind::object FromIterableTree(nanobind::handle xs) const;

  int num_leaves() const {
    if (traversal_.empty()) {
      return 0;
    }
    return traversal_.back().num_leaves;
  }

  int num_nodes() const { return traversal_.size(); }

  PyTreeRegistry* registry() const { return registry_; }

  size_t Hash() const;

  bool operator==(const PyTreeDef& other) const;
  bool operator!=(const PyTreeDef& other) const { return !(*this == other); }

  std::string ToString() const;

  // Transforms the PyTreeDef into a pickleable object. Used to implement
  // `PyTreeDef.__getstate__`.
  nanobind::object ToPickle() const;

  // Transforms the object returned by `ToPickleable()` back to PyTreeDef. Used
  // to implement `PyTreeDef.__setstate__`.
  void FromPickle(nanobind::object pickleable);

  void SerializeTo(jax::PyTreeDefProto& result) const;

  static nb_class_ptr<PyTreeDef> DeserializeFrom(
      nb_class_ptr<PyTreeRegistry> registry, const jax::PyTreeDefProto& input);

  std::optional<std::pair<nanobind::object, nanobind::object>> GetNodeData()
      const;

  static nb_class_ptr<PyTreeDef> MakeFromNodeDataAndChildren(
      nb_class_ptr<PyTreeRegistry> registry,
      std::optional<std::pair<nanobind::object, nanobind::object>> node_data,
      nanobind::iterable children);

  static PyType_Slot slots_[];

 private:
  void SetNumLeavesAndNumNodes();

  struct Node {
    PyTreeKind kind = PyTreeKind::kLeaf;

    // Arity for non-kLeaf types.
    int arity = 0;

    // Kind-specific auxiliary data. For a kNamedTuple, contains the tuple type
    // object. For a kDict, use `sorted_dict_keys` field below. For a kCustom
    // type, contains the auxiliary data returned by the `to_iterable` function.
    nanobind::object node_data;

    // Kind-specific auxiliary data specialized for kDict. Use a c++ vector
    // to hold the sorted dict keys instead of a py::list to avoid creating
    // a new python list object when flattening kDict. For deeply nested dict,
    // using c++ vector instead of py::list avoids creating too many python
    // objects that make python gc sweep slow.
    std::vector<nanobind::object> sorted_dict_keys;

    // Custom type registration. Must be null for non-custom types.
    const PyTreeRegistry::Registration* custom = nullptr;

    // Number of leaf nodes in the subtree rooted at this node.
    int num_leaves = 0;

    // Number of leaf and interior nodes in the subtree rooted at this node.
    int num_nodes = 0;

    int tp_traverse(visitproc visit, void* arg) const;
  };
  template <typename H>
  friend H AbslHashValue(H h, const Node& n);

  template <typename H>
  friend H AbslHashValue(H h, const PyTreeDef& t);

  // Helper that manufactures an instance of a node given its children.
  static nanobind::object MakeNode(const Node& node,
                                   absl::Span<nanobind::object> children);

  // Recursive helper used to implement FromIterableTree()
  nanobind::object FromIterableTreeHelper(
      nanobind::handle xs,
      absl::InlinedVector<PyTreeDef::Node, 1>::const_reverse_iterator* it)
      const;

  template <typename T>
  void FlattenImpl(nanobind::handle handle, T& leaves,
                   const std::optional<nanobind::callable>& leaf_predicate,
                   std::optional<std::vector<nanobind::object>>& keypath);

  template <typename T>
  nanobind::object UnflattenImpl(T leaves) const;

  static int tp_traverse(PyObject* self, visitproc visit, void* arg);
  static int tp_clear(PyObject* self);

  // Pytree registry. Not owned.
  PyTreeRegistry* registry_;
  // If this class holds a reference to `registry`, it is held by
  // `registry_ref_`.
  nb_class_ptr<PyTreeRegistry> registry_ref_;

  // Nodes, in a post-order traversal. We use an ordered traversal to minimize
  // allocations, and post-order corresponds to the order we need to rebuild the
  // tree structure.
  absl::InlinedVector<Node, 1> traversal_;
};

template <typename H>
H AbslHashValue(H h, const PyTreeDef::Node& n) {
  h = H::combine(std::move(h), n.kind, n.arity, n.custom);
  return h;
}

template <typename H>
H AbslHashValue(H h, const PyTreeDef& t) {
  h = H::combine(std::move(h), t.traversal_);
  return h;
}

void BuildPytreeSubmodule(nanobind::module_& m);

}  // namespace xla

#endif  // XLA_PYTHON_PYTREE_H_
