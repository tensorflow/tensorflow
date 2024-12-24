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

// Caution: this code uses exceptions. The exception use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#include "xla/python/pytree.h"

#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/pjrt/exceptions.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/pytree.pb.h"
#include "tsl/platform/logging.h"

namespace xla {

namespace nb = nanobind;

constexpr int kSequenceKeyHashSalt = 1;
constexpr int kFlattenedIndexKeyHashSalt = 42;

PyTreeRegistry::PyTreeRegistry(bool enable_none, bool enable_tuple,
                               bool enable_namedtuple, bool enable_list,
                               bool enable_dict) {
  auto add_builtin_type = [&](PyTypeObject* type_obj, PyTreeKind kind) {
    nb::object type =
        nb::borrow<nb::object>(reinterpret_cast<PyObject*>(type_obj));
    auto registration = std::make_unique<Registration>();
    registration->kind = kind;
    registration->type = type;
    CHECK(registrations_.emplace(type, std::move(registration)).second);
  };
  if (enable_none) {
    add_builtin_type(Py_TYPE(Py_None), PyTreeKind::kNone);
  }
  if (enable_tuple) {
    add_builtin_type(&PyTuple_Type, PyTreeKind::kTuple);
  }
  enable_namedtuple_ = enable_namedtuple;
  if (enable_list) {
    add_builtin_type(&PyList_Type, PyTreeKind::kList);
  }
  if (enable_dict) {
    add_builtin_type(&PyDict_Type, PyTreeKind::kDict);
  }
}

void PyTreeRegistry::Register(
    nb::object type, nb::callable to_iterable, nb::callable from_iterable,
    std::optional<nb::callable> to_iterable_with_keys) {
  auto registration = std::make_unique<Registration>();
  registration->kind = PyTreeKind::kCustom;
  registration->type = type;
  registration->to_iterable = std::move(to_iterable);
  registration->from_iterable = std::move(from_iterable);
  registration->to_iterable_with_keys = std::move(to_iterable_with_keys);
  auto it = registrations_.emplace(type, std::move(registration));
  if (!it.second) {
    throw std::invalid_argument(
        absl::StrFormat("Duplicate custom PyTreeDef type registration for %s.",
                        nb::cast<absl::string_view>(nb::repr(type))));
  }
}

void PyTreeRegistry::RegisterDataclass(nb::object type,
                                       std::vector<nb::str> data_fields,
                                       std::vector<nb::str> meta_fields) {
  auto registration = std::make_unique<Registration>();
  registration->kind = PyTreeKind::kDataclass;
  registration->type = type;
  registration->data_fields = std::move(data_fields);
  registration->meta_fields = std::move(meta_fields);
  auto it = registrations_.emplace(type, std::move(registration));
  if (!it.second) {
    throw std::invalid_argument(absl::StrFormat(
        "Duplicate custom dataclass PyTreeDef type registration for %s.",
        nb::cast<absl::string_view>(nb::repr(std::move(type)))));
  }
}

std::pair<nanobind::iterable, nanobind::object>
PyTreeRegistry::Registration::ToIterable(nanobind::handle o) const {
  nb::object out = to_iterable(o);
  nb::tuple leaves_and_aux_data;
  if (!nb::try_cast<nb::tuple>(out, leaves_and_aux_data) ||
      leaves_and_aux_data.size() != 2) {
    throw std::invalid_argument(absl::StrCat(
        "The to_iterable function for a custom PyTree node should return "
        "a (children, aux_data) tuple, got ",
        nb::cast<absl::string_view>(nb::repr(out))));
  }
  nb::iterable leaves;
  if (!nb::try_cast<nb::iterable>(leaves_and_aux_data[0], leaves)) {
    throw std::invalid_argument(absl::StrCat(
        "The to_iterable function for a custom PyTree node should return "
        "a (children, aux_data) tuple where 'children' is iterable, "
        "got ",
        nb::cast<absl::string_view>(nb::repr(out))));
  }
  return std::make_pair(std::move(leaves), nb::object(leaves_and_aux_data[1]));
}

std::pair<std::vector<std::pair<nb::object, nb::object>>, nb::object>
PyTreeRegistry::Registration::ToIterableWithKeys(nb::handle o) const {
  // Backwards compatibility case: return dummy FlattenedIndexKey for each leaf.
  std::vector<std::pair<nb::object, nb::object>> result;
  if (!to_iterable_with_keys.has_value()) {
    auto [leaves, aux_data] = ToIterable(o);
    for (nb::handle leaf : leaves) {
      result.push_back(std::make_pair(
          make_nb_class<FlattenedIndexKey>(result.size()), nb::borrow(leaf)));
    }
    return std::make_pair(std::move(result), std::move(aux_data));
  }
  nb::object out = to_iterable_with_keys.value()(o);
  nb::tuple leaves_and_aux_data;
  if (!nb::try_cast<nb::tuple>(out, leaves_and_aux_data) ||
      leaves_and_aux_data.size() != 2) {
    throw std::invalid_argument(absl::StrCat(
        "The to_iterable_with_keys function for a custom PyTree "
        "node should return a (key_leaf_pairs, aux_data) tuple, got ",
        nb::cast<absl::string_view>(nb::repr(out))));
  }
  nb::iterable key_leaf_pairs;
  if (!nb::try_cast<nb::iterable>(leaves_and_aux_data[0], key_leaf_pairs)) {
    throw std::invalid_argument(absl::StrCat(
        "The to_iterable_with_keys function for a custom PyTree node should "
        "return a (key_leaf_pairs, aux_data) tuple where 'key_leaf_pairs' is "
        "iterable, got ",
        nb::cast<absl::string_view>(nb::repr(leaves_and_aux_data))));
  }
  for (nb::handle key_leaf_pair : key_leaf_pairs) {
    nb::tuple key_leaf_pair_tuple;
    if (!nb::try_cast<nb::tuple>(key_leaf_pair, key_leaf_pair_tuple) ||
        key_leaf_pair_tuple.size() != 2) {
      throw std::invalid_argument(absl::StrCat(
          "The to_iterable_with_keys function for a custom PyTree node should "
          "return a (key_leaf_pairs, aux_data) tuple where 'child",
          nb::cast<absl::string_view>(nb::repr(key_leaf_pair))));
    }
    result.push_back(std::make_pair(nb::borrow(key_leaf_pair_tuple[0]),
                                    nb::borrow(key_leaf_pair_tuple[1])));
  }
  return std::make_pair(std::move(result), nb::object(leaves_and_aux_data[1]));
}

int PyTreeRegistry::Registration::tp_traverse(visitproc visit, void* arg) {
  Py_VISIT(type.ptr());
  Py_VISIT(to_iterable.ptr());
  Py_VISIT(from_iterable.ptr());
  for (const auto& field : data_fields) {
    Py_VISIT(field.ptr());
  }
  for (const auto& field : meta_fields) {
    Py_VISIT(field.ptr());
  }
  return 0;
}

// Computes the node kind of a given Python object.
PyTreeKind PyTreeRegistry::KindOfObject(
    nb::handle obj, PyTreeRegistry::Registration const** custom) const {
  const PyTreeRegistry::Registration* registration = Lookup(obj.type());
  if (registration) {
    if (registration->kind == PyTreeKind::kCustom ||
        registration->kind == PyTreeKind::kDataclass) {
      *custom = registration;
    } else {
      *custom = nullptr;
    }
    return registration->kind;
  } else if (nb::isinstance<nb::tuple>(obj) && nb::hasattr(obj, "_fields")) {
    // We can only identify namedtuples heuristically, here by the presence of
    // a _fields attribute.
    return PyTreeKind::kNamedTuple;
  } else {
    return PyTreeKind::kLeaf;
  }
}

/*static*/ const PyTreeRegistry::Registration* PyTreeRegistry::Lookup(
    nb::handle type) const {
  auto it = registrations_.find(type);
  return it == registrations_.end() ? nullptr : it->second.get();
}

/*static*/ std::vector<nb::object> GetSortedPyDictKeys(PyObject* py_dict) {
  std::vector<nb::object> keys;
  keys.reserve(PyDict_Size(py_dict));
  PyObject* key;
  Py_ssize_t pos = 0;
  while (PyDict_Next(py_dict, &pos, &key, /*value=*/nullptr)) {
    keys.push_back(nb::borrow<nb::object>(key));
  }

  try {
    std::stable_sort(
        keys.begin(), keys.end(), [](const nb::object& a, const nb::object& b) {
          int cmp = PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_LT);
          if (cmp == -1) {
            throw nb::python_error();
          }
          return cmp;
        });
  } catch (nb::python_error& e) {
    nb::raise_from(e, PyExc_ValueError,
                   "Comparator raised exception while sorting pytree "
                   "dictionary keys.");
  }
  return keys;
}

/*static*/ bool IsSortedPyDictKeysEqual(absl::Span<const nb::object> lhs,
                                        absl::Span<const nb::object> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (lhs[i].not_equal(rhs[i])) {
      return false;
    }
  }
  return true;
}

bool PyTreeDef::operator==(const PyTreeDef& other) const {
  if (traversal_.size() != other.traversal_.size()) {
    return false;
  }
  for (size_t i = 0; i < traversal_.size(); ++i) {
    const Node& a = traversal_[i];
    const Node& b = other.traversal_[i];
    if (a.kind != b.kind || a.arity != b.arity ||
        (a.node_data.ptr() == nullptr) != (b.node_data.ptr() == nullptr) ||
        (a.sorted_dict_keys.size() != b.sorted_dict_keys.size()) ||
        a.custom != b.custom) {
      return false;
    }
    if (a.node_data && a.node_data.not_equal(b.node_data)) {
      return false;
    }
    if (!IsSortedPyDictKeysEqual(a.sorted_dict_keys, b.sorted_dict_keys)) {
      return false;
    }
    // We don't need to test equality of num_leaves and num_nodes since they
    // are derivable from the other node data.
  }
  return true;
}

nb::object PyTreeRegistry::FlattenOneLevel(nb::handle x) const {
  return FlattenOneLevelImpl(x, /*with_keys=*/false);
}

nb::object PyTreeRegistry::FlattenOneLevelWithKeys(nb::handle x) const {
  return FlattenOneLevelImpl(x, /*with_keys=*/true);
}

nb::object PyTreeRegistry::FlattenOneLevelImpl(nb::handle x,
                                               bool with_keys) const {
  PyTreeRegistry::Registration const* custom;
  PyTreeKind kind = KindOfObject(x, &custom);
  switch (kind) {
    case PyTreeKind::kNone:
      return nb::make_tuple(nb::make_tuple(), nb::none());
    case PyTreeKind::kTuple: {
      if (with_keys) {
        auto size = PyTuple_GET_SIZE(x.ptr());
        nb::object key_leaves = nb::steal(PyTuple_New(size));
        for (int i = 0; i < size; ++i) {
          nb::object key = make_nb_class<SequenceKey>(i);
          nb::object value =
              nb::borrow<nb::object>(PyTuple_GET_ITEM(x.ptr(), i));
          PyTuple_SET_ITEM(key_leaves.ptr(), i,
                           nb::make_tuple(key, value).release().ptr());
        }
        return nb::make_tuple(std::move(key_leaves), nb::none());
      }
      return nb::make_tuple(nb::borrow(x), nb::none());
    }
    case PyTreeKind::kList: {
      if (with_keys) {
        auto size = PyList_GET_SIZE(x.ptr());
        nb::object key_leaves = nb::steal(PyTuple_New(size));
        for (int i = 0; i < size; ++i) {
          nb::object key = make_nb_class<SequenceKey>(i);
          nb::object value =
              nb::borrow<nb::object>(PyList_GET_ITEM(x.ptr(), i));
          PyTuple_SET_ITEM(key_leaves.ptr(), i,
                           nb::make_tuple(key, value).release().ptr());
        }
        return nb::make_tuple(std::move(key_leaves), nb::none());
      }
      return nb::make_tuple(nb::borrow(x), nb::none());
    }
    case PyTreeKind::kDict: {
      nb::dict dict = nb::borrow<nb::dict>(x);
      std::vector<nb::object> sorted_keys = GetSortedPyDictKeys(dict.ptr());
      nb::tuple keys = nb::steal<nb::tuple>(PyTuple_New(sorted_keys.size()));
      nb::tuple values = nb::steal<nb::tuple>(PyTuple_New(sorted_keys.size()));
      for (size_t i = 0; i < sorted_keys.size(); ++i) {
        nb::object& key = sorted_keys[i];
        nb::object value = nb::object(dict[key]);
        if (with_keys) {
          value = nb::make_tuple(make_nb_class<DictKey>(key), value);
        }
        PyTuple_SET_ITEM(values.ptr(), i, value.release().ptr());
        PyTuple_SET_ITEM(keys.ptr(), i, sorted_keys[i].release().ptr());
      }
      return nb::make_tuple(std::move(values), std::move(keys));
    }
    case PyTreeKind::kNamedTuple: {
      nb::tuple in = nb::borrow<nb::tuple>(x);
      nb::list out;
      if (with_keys) {
        // Get key names from NamedTuple fields.
        nb::tuple fields;
        if (!nb::try_cast<nb::tuple>(nb::getattr(in, "_fields"), fields) ||
            in.size() != fields.size()) {
          throw std::invalid_argument(
              "A namedtuple's _fields attribute should have the same size as "
              "the tuple.");
        }
        auto field_iter = fields.begin();
        for (nb::handle entry : in) {
          out.append(nb::make_tuple(
              make_nb_class<GetAttrKey>(nb::str(*field_iter)), entry));
        }
        return nb::make_tuple(std::move(out), x.type());
      }
      for (size_t i = 0; i < in.size(); ++i) {
        out.append(in[i]);
      }
      return nb::make_tuple(std::move(out), x.type());
    }
    case PyTreeKind::kCustom: {
      if (with_keys) {
        auto [leaves, aux_data] = custom->ToIterableWithKeys(x);
        return nb::make_tuple(std::move(leaves), std::move(aux_data));
      }
      auto [leaves, aux_data] = custom->ToIterable(x);
      return nb::make_tuple(std::move(leaves), std::move(aux_data));
    }
    case PyTreeKind::kDataclass: {
      auto data_size = custom->data_fields.size();
      nb::list leaves = nb::steal<nb::list>(PyList_New(data_size));
      for (int leaf = 0; leaf < data_size; ++leaf) {
        nb::object value = nb::getattr(x, custom->data_fields[leaf]);
        if (with_keys) {
          value = nb::make_tuple(
              make_nb_class<GetAttrKey>(custom->data_fields[leaf]), value);
        }
        PyList_SET_ITEM(leaves.ptr(), leaf, value.release().ptr());
      }
      auto meta_size = custom->meta_fields.size();
      nb::object aux_data = nb::steal(PyTuple_New(meta_size));
      for (int meta_leaf = 0; meta_leaf < meta_size; ++meta_leaf) {
        PyTuple_SET_ITEM(
            aux_data.ptr(), meta_leaf,
            nb::getattr(x, custom->meta_fields[meta_leaf]).release().ptr());
      }
      return nb::make_tuple(std::move(leaves), std::move(aux_data));
    }
    default:
      DCHECK(kind == PyTreeKind::kLeaf);
      return nb::none();
  }
}

/* static */ PyType_Slot PyTreeRegistry::slots_[] = {
    {Py_tp_traverse, (void*)PyTreeRegistry::tp_traverse},
    {Py_tp_clear, (void*)PyTreeRegistry::tp_clear},
    {0, nullptr},
};

/* static */ int PyTreeRegistry::tp_traverse(PyObject* self, visitproc visit,
                                             void* arg) {
  PyTreeRegistry* registry = nb::inst_ptr<PyTreeRegistry>(self);
  Py_VISIT(Py_TYPE(self));
  for (const auto& [key, value] : registry->registrations_) {
    Py_VISIT(key.ptr());
    int rval = value->tp_traverse(visit, arg);
    if (rval != 0) {
      return rval;
    }
  }
  return 0;
}

/* static */ int PyTreeRegistry::tp_clear(PyObject* self) {
  PyTreeRegistry* registry = nb::inst_ptr<PyTreeRegistry>(self);
  registry->registrations_.clear();
  return 0;
}

/* static */ PyType_Slot DictKey::slots_[] = {
    {Py_tp_traverse, (void*)DictKey::tp_traverse},
    {Py_tp_clear, (void*)DictKey::tp_clear},
    {0, nullptr},
};

/* static */ int DictKey::tp_traverse(PyObject* self, visitproc visit,
                                      void* arg) {
  DictKey* key = nb::inst_ptr<DictKey>(self);
  Py_VISIT(key->key_.ptr());
  return 0;
}

/* static */ int DictKey::tp_clear(PyObject* self) {
  DictKey* dictkey = nb::inst_ptr<DictKey>(self);
  nb::object tmp;
  std::swap(tmp, dictkey->key_);
  return 0;
}

std::string SequenceKey::ToString() const {
  return absl::StrFormat("[%d]", idx_);
}

std::string SequenceKey::ToReprString() const {
  return absl::StrFormat("SequenceKey(idx=%d)", idx_);
}

std::string DictKey::ToString() const {
  return absl::StrFormat("[%s]", nb::cast<absl::string_view>(nb::repr(key_)));
}

std::string DictKey::ToReprString() const {
  return absl::StrFormat("DictKey(key=%s)",
                         nb::cast<absl::string_view>(nb::repr(key_)));
}

std::string GetAttrKey::ToString() const {
  return absl::StrFormat(".%s", nb::cast<absl::string_view>(name_));
}

std::string GetAttrKey::ToReprString() const {
  return absl::StrFormat("GetAttrKey(name='%s')",
                         nb::cast<absl::string_view>(name_));
}

std::string FlattenedIndexKey::ToString() const {
  return absl::StrFormat("[<flat index %d>]", key_);
}

std::string FlattenedIndexKey::ToReprString() const {
  return absl::StrFormat("FlattenedIndexKey(key=%d)", key_);
}

bool SequenceKey::Equals(const nb::object& other) {
  SequenceKey other_key(0);
  if (!nb::try_cast<SequenceKey>(other, other_key)) return false;
  return idx_ == other_key.idx();
}

bool DictKey::Equals(const nb::object& other) {
  DictKey other_key(nb::none());
  if (!nb::try_cast<DictKey>(other, other_key)) return false;
  return key_.equal(other_key.key());
}

bool GetAttrKey::Equals(const nb::object& other) {
  GetAttrKey other_key(nb::str(""));
  if (!nb::try_cast<GetAttrKey>(other, other_key)) return false;
  return name_.equal(other_key.name());
}

bool FlattenedIndexKey::Equals(const nb::object& other) {
  FlattenedIndexKey other_key(0);
  if (!nb::try_cast<FlattenedIndexKey>(other, other_key)) return false;
  return key_ == other_key.key();
}

nanobind::tuple SequenceKey::MatchArgs(nanobind::handle unused) {
  return nanobind::make_tuple("idx");
};

nanobind::tuple DictKey::MatchArgs(nanobind::handle unused) {
  return nanobind::make_tuple("key");
};

nanobind::tuple GetAttrKey::MatchArgs(nanobind::handle unused) {
  return nanobind::make_tuple("name");
};

nanobind::tuple FlattenedIndexKey::MatchArgs(nanobind::handle unused) {
  return nanobind::make_tuple("key");
};

template <typename T>
void PyTreeDef::FlattenImpl(nb::handle handle, T& leaves,
                            const std::optional<nb::callable>& leaf_predicate,
                            std::optional<std::vector<nb::object>>& keypath) {
  Node node;
  const int start_num_nodes = traversal_.size();
  const int start_num_leaves = leaves.size();
  bool is_known_leaf = false;
  if (leaf_predicate) {
    nb::object o = (*leaf_predicate)(handle);
    // Historically we accepted "truthy" values from leaf predicates. Accept
    // None here to keep existing clients happy.
    if (o.is_none()) {
      is_known_leaf = false;
    } else if (!nb::try_cast<bool>(o, is_known_leaf)) {
      throw std::invalid_argument(absl::StrCat(
          "is_leaf predicate returned a non-boolean value ",
          nb::cast<absl::string_view>(nb::repr(o)), "; expected a boolean"));
    }
  }
  if (is_known_leaf) {
    nb::object value = nb::borrow<nb::object>(handle);
    if (keypath.has_value()) {
      const std::vector<nb::object>& frozen_keypath = keypath.value();
      nb::object kp_tuple = nb::steal(PyTuple_New(frozen_keypath.size()));
      for (int i = 0; i < frozen_keypath.size(); ++i) {
        PyTuple_SET_ITEM(kp_tuple.ptr(), i,
                         nb::object(frozen_keypath[i]).release().ptr());
      }
      value = nb::make_tuple(std::move(kp_tuple), std::move(value));
    }
    if constexpr (std::is_same_v<T, nb::list>) {
      leaves.append(std::move(value));
    } else {
      leaves.push_back(std::move(value));
    }
  } else {
    node.kind = registry_->KindOfObject(handle, &node.custom);
    auto recurse = [this, &leaf_predicate, &leaves](
                       nb::handle child,
                       std::optional<std::vector<nb::object>>& keypath) {
      if (Py_EnterRecursiveCall(
              " in flatten; PyTree may have cyclical node references.")) {
        return;
      }
      FlattenImpl(child, leaves, leaf_predicate, keypath);
      Py_LeaveRecursiveCall();
    };
    switch (node.kind) {
      case PyTreeKind::kNone:
        // Nothing to do.
        break;
      case PyTreeKind::kTuple: {
        node.arity = PyTuple_GET_SIZE(handle.ptr());
        for (int i = 0; i < node.arity; ++i) {
          if (keypath.has_value()) {
            keypath->push_back(make_nb_class<SequenceKey>(i));
          }
          recurse(PyTuple_GET_ITEM(handle.ptr(), i), keypath);
          if (keypath.has_value()) {
            keypath->pop_back();
          }
        }
        break;
      }
      case PyTreeKind::kList: {
        node.arity = PyList_GET_SIZE(handle.ptr());
        for (int i = 0; i < node.arity; ++i) {
          if (keypath.has_value()) {
            keypath->push_back(make_nb_class<SequenceKey>(i));
          }
          recurse(PyList_GET_ITEM(handle.ptr(), i), keypath);
          if (keypath.has_value()) {
            keypath->pop_back();
          }
        }
        break;
      }
      case PyTreeKind::kDict: {
        nb::dict dict = nb::borrow<nb::dict>(handle);

        std::vector<nb::object> keys = GetSortedPyDictKeys(dict.ptr());
        for (nb::object& key : keys) {
          if (keypath.has_value()) {
            keypath->push_back(make_nb_class<DictKey>(key));
          }
          recurse(dict[key], keypath);
          if (keypath.has_value()) {
            keypath->pop_back();
          }
        }
        node.arity = dict.size();
        node.sorted_dict_keys = std::move(keys);
        break;
      }
      case PyTreeKind::kCustom: {
        if (keypath.has_value()) {
          auto [leaves, aux_data] = node.custom->ToIterableWithKeys(handle);
          node.node_data = std::move(aux_data);
          node.arity = 0;
          for (auto& [key, leaf] : leaves) {
            keypath->push_back(key);
            ++node.arity;
            recurse(leaf, keypath);
            keypath->pop_back();
          }
        } else {
          auto [leaves, aux_data] = node.custom->ToIterable(handle);
          node.node_data = std::move(aux_data);
          node.arity = 0;
          for (nb::handle entry : leaves) {
            ++node.arity;
            recurse(entry, keypath);
          }
        }
        break;
      }
      case PyTreeKind::kDataclass: {
        auto meta_size = node.custom->meta_fields.size();
        nb::object aux_data = nb::steal(PyTuple_New(meta_size));
        for (int meta_leaf = 0; meta_leaf < meta_size; ++meta_leaf) {
          PyTuple_SET_ITEM(
              aux_data.ptr(), meta_leaf,
              nb::getattr(handle, node.custom->meta_fields[meta_leaf])
                  .release()
                  .ptr());
        }
        node.node_data = std::move(aux_data);
        auto data_size = node.custom->data_fields.size();
        node.arity = data_size;
        for (int leaf = 0; leaf < data_size; ++leaf) {
          if (keypath.has_value()) {
            keypath->push_back(
                make_nb_class<GetAttrKey>(node.custom->data_fields[leaf]));
          }
          recurse(nb::getattr(handle, node.custom->data_fields[leaf]), keypath);
          if (keypath.has_value()) {
            keypath->pop_back();
          }
        }
        break;
      }
      case PyTreeKind::kNamedTuple: {
        nb::tuple tuple = nb::borrow<nb::tuple>(handle);
        node.arity = tuple.size();
        node.node_data = nb::borrow<nb::object>(tuple.type());
        if (keypath.has_value()) {
          // Get key names from NamedTuple fields.
          nb::tuple fields;
          if (!nb::try_cast<nb::tuple>(nb::getattr(tuple, "_fields"), fields) ||
              tuple.size() != fields.size()) {
            throw std::invalid_argument(
                "A namedtuple's _fields attribute should have the same size as "
                "the tuple.");
          }
          auto field_iter = fields.begin();
          for (nb::handle entry : tuple) {
            keypath->push_back(make_nb_class<GetAttrKey>(nb::str(*field_iter)));
            field_iter++;
            recurse(entry, keypath);
            keypath->pop_back();
          }
        } else {
          for (nb::handle entry : tuple) {
            recurse(entry, keypath);
          }
        }
        break;
      }
      default:
        DCHECK(node.kind == PyTreeKind::kLeaf);
        auto value = nb::borrow<nb::object>(handle);
        if (keypath.has_value()) {
          const std::vector<nb::object>& frozen_keypath = keypath.value();
          nb::object kp_tuple = nb::steal(PyTuple_New(frozen_keypath.size()));
          for (int i = 0; i < frozen_keypath.size(); ++i) {
            PyTuple_SET_ITEM(kp_tuple.ptr(), i,
                             nb::object(frozen_keypath[i]).release().ptr());
          }
          value = nb::make_tuple(std::move(kp_tuple), std::move(value));
        }
        if constexpr (std::is_same_v<T, nb::list>) {
          leaves.append(std::move(value));
        } else {
          leaves.push_back(std::move(value));
        }
    }
  }
  node.num_nodes = traversal_.size() - start_num_nodes + 1;
  node.num_leaves = leaves.size() - start_num_leaves;
  traversal_.push_back(std::move(node));
}

void PyTreeDef::Flatten(nb::handle handle,
                        absl::InlinedVector<nb::object, 2>& leaves,
                        std::optional<nb::callable> leaf_predicate) {
  std::optional<std::vector<nb::object>> keypath = std::nullopt;
  FlattenImpl(handle, leaves, leaf_predicate, keypath);
}

void PyTreeDef::Flatten(nb::handle handle, std::vector<nb::object>& leaves,
                        std::optional<nb::callable> leaf_predicate) {
  std::optional<std::vector<nb::object>> keypath = std::nullopt;
  FlattenImpl(handle, leaves, leaf_predicate, keypath);
}

void PyTreeDef::Flatten(nb::handle handle, nb::list& leaves,
                        std::optional<nb::callable> leaf_predicate) {
  std::optional<std::vector<nb::object>> keypath = std::nullopt;
  FlattenImpl(handle, leaves, leaf_predicate, keypath);
}

/*static*/ std::pair<std::vector<nb::object>, nb_class_ptr<PyTreeDef>>
PyTreeDef::Flatten(nb::handle x, nb_class_ptr<PyTreeRegistry> registry,
                   std::optional<nb::callable> leaf_predicate) {
  auto def = make_nb_class<PyTreeDef>(registry);
  std::vector<nb::object> leaves;
  def->Flatten(x, leaves, leaf_predicate);
  return std::make_pair(std::move(leaves), std::move(def));
}

void PyTreeDef::FlattenWithPath(nb::handle handle, nanobind::list& leaves,
                                std::optional<nb::callable> leaf_predicate) {
  std::optional<std::vector<nb::object>> keypath = std::vector<nb::object>();
  FlattenImpl(handle, leaves, leaf_predicate, keypath);
}

/*static*/ bool PyTreeDef::AllLeaves(PyTreeRegistry* registry,
                                     const nb::iterable& x) {
  const PyTreeRegistry::Registration* custom;
  for (const nb::handle& h : x) {
    if (registry->KindOfObject(h, &custom) != PyTreeKind::kLeaf) return false;
  }
  return true;
}

template <typename T>
nb::object PyTreeDef::UnflattenImpl(T leaves) const {
  absl::InlinedVector<nb::object, 4> agenda;
  auto it = leaves.begin();
  int leaf_count = 0;
  for (const Node& node : traversal_) {
    if (agenda.size() < node.arity) {
      throw std::logic_error("Too few elements for TreeDef node.");
    }
    switch (node.kind) {
      case PyTreeKind::kLeaf:
        if (it == leaves.end()) {
          throw std::invalid_argument(absl::StrFormat(
              "Too few leaves for PyTreeDef; expected %d, got %d", num_leaves(),
              leaf_count));
        }
        agenda.push_back(nb::borrow<nb::object>(*it));
        ++it;
        ++leaf_count;
        break;

      case PyTreeKind::kNone:
      case PyTreeKind::kTuple:
      case PyTreeKind::kNamedTuple:
      case PyTreeKind::kList:
      case PyTreeKind::kDict:
      case PyTreeKind::kCustom:
      case PyTreeKind::kDataclass: {
        const int size = agenda.size();
        absl::Span<nb::object> span;
        if (node.arity > 0) {
          span = absl::Span<nb::object>(&agenda[size - node.arity], node.arity);
        }
        nb::object o = MakeNode(node, span);
        agenda.resize(size - node.arity);
        agenda.push_back(o);
        break;
      }
    }
  }
  if (it != leaves.end()) {
    throw std::invalid_argument(absl::StrFormat(
        "Too many leaves for PyTreeDef; expected %d.", num_leaves()));
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return std::move(agenda.back());
}

nb::object PyTreeDef::Unflatten(nb::iterable leaves) const {
  return UnflattenImpl(leaves);
}

nb::object PyTreeDef::Unflatten(absl::Span<const nb::object> leaves) const {
  return UnflattenImpl(leaves);
}

/*static*/ nb::object PyTreeDef::MakeNode(const PyTreeDef::Node& node,
                                          absl::Span<nb::object> children) {
  if (children.size() != node.arity) {
    throw std::logic_error("Node arity mismatch.");
  }
  switch (node.kind) {
    case PyTreeKind::kLeaf:
      throw std::logic_error("MakeNode not implemented for leaves.");

    case PyTreeKind::kNone:
      return nb::none();

    case PyTreeKind::kTuple:
    case PyTreeKind::kNamedTuple: {
      nb::object tuple = nb::steal(PyTuple_New(node.arity));
      for (int i = 0; i < node.arity; ++i) {
        PyTuple_SET_ITEM(tuple.ptr(), i, children[i].release().ptr());
      }
      if (node.kind == PyTreeKind::kNamedTuple) {
        return node.node_data(*tuple);
      } else {
        return tuple;
      }
    }

    case PyTreeKind::kList: {
      nb::object list = nb::steal(PyList_New(node.arity));
      for (int i = 0; i < node.arity; ++i) {
        PyList_SET_ITEM(list.ptr(), i, children[i].release().ptr());
      }
      return list;
    }

    case PyTreeKind::kDict: {
      nb::dict dict;
      for (int i = 0; i < node.arity; ++i) {
        dict[node.sorted_dict_keys[i]] = std::move(children[i]);
      }
      return std::move(dict);
      break;
    }
    case PyTreeKind::kCustom: {
      nb::object tuple = nb::steal(PyTuple_New(node.arity));
      for (int i = 0; i < node.arity; ++i) {
        PyTuple_SET_ITEM(tuple.ptr(), i, children[i].release().ptr());
      }
      return node.custom->from_iterable(node.node_data, tuple);
    }

    case PyTreeKind::kDataclass: {
      nb::kwargs kwargs;
      auto meta_size = node.custom->meta_fields.size();
      for (int i = 0; i < meta_size; ++i) {
        kwargs[node.custom->meta_fields[i]] =
            nb::borrow(nb::tuple(node.node_data)[i]);
      }
      auto data_size = node.custom->data_fields.size();
      for (int i = 0; i < data_size; ++i) {
        kwargs[node.custom->data_fields[i]] = std::move(children[i]);
      }
      return node.custom->type(**kwargs);
    }
  }
  throw std::logic_error("Unreachable code.");
}

nb::list PyTreeDef::FlattenUpTo(nb::handle xs) const {
  nb::list leaves = nb::steal<nb::list>(PyList_New(num_leaves()));
  std::vector<nb::object> agenda;
  agenda.push_back(nb::borrow<nb::object>(xs));
  auto it = traversal_.rbegin();
  int leaf = num_leaves() - 1;
  while (!agenda.empty()) {
    if (it == traversal_.rend()) {
      throw std::invalid_argument(absl::StrFormat(
          "Tree structures did not match: %s vs %s",
          nb::cast<absl::string_view>(nb::repr(xs)), ToString()));
    }
    const Node& node = *it;
    nb::object object = agenda.back();
    agenda.pop_back();
    ++it;

    switch (node.kind) {
      case PyTreeKind::kLeaf:
        if (leaf < 0) {
          throw std::logic_error("Leaf count mismatch.");
        }
        PyList_SET_ITEM(leaves.ptr(), leaf, object.release().ptr());
        --leaf;
        break;

      case PyTreeKind::kNone:
        if (!object.is_none()) {
          throw std::invalid_argument(absl::StrFormat(
              "Expected None, got %s.\n\n"
              "In previous releases of JAX, flatten-up-to used to "
              "consider None to be a tree-prefix of non-None values. To obtain "
              "the previous behavior, you can usually write:\n"
              "  jax.tree.map(lambda x, y: None if x is None else f(x, y), a, "
              "b, is_leaf=lambda x: x is None)",
              nb::cast<absl::string_view>(nb::repr(object))));
        }
        break;

      case PyTreeKind::kTuple: {
        if (!PyTuple_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected tuple, got %s.",
                              nb::cast<absl::string_view>(nb::repr(object))));
        }
        nb::tuple tuple = nb::borrow<nb::tuple>(object);
        if (tuple.size() != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "Tuple arity mismatch: %d != %d; tuple: %s.", tuple.size(),
              node.arity, nb::cast<absl::string_view>(nb::repr(object))));
        }
        for (nb::handle entry : tuple) {
          agenda.push_back(nb::borrow<nb::object>(entry));
        }
        break;
      }

      case PyTreeKind::kList: {
        if (!PyList_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected list, got %s.",
                              nb::cast<absl::string_view>(nb::repr(object))));
        }
        nb::list list = nb::borrow<nb::list>(object);
        if (list.size() != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "List arity mismatch: %d != %d; list: %s.", list.size(),
              node.arity, nb::cast<absl::string_view>(nb::repr(object))));
        }
        for (nb::handle entry : list) {
          agenda.push_back(nb::borrow<nb::object>(entry));
        }
        break;
      }

      case PyTreeKind::kDict: {
        if (!PyDict_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected dict, got %s.",
                              nb::cast<absl::string_view>(nb::repr(object))));
        }
        nb::dict dict = nb::borrow<nb::dict>(object);
        std::vector<nb::object> keys = GetSortedPyDictKeys(dict.ptr());
        if (!IsSortedPyDictKeysEqual(keys, node.sorted_dict_keys)) {
          // Convert to a nb::list for nb::repr to avoid having to stringify a
          // vector. This is error path so it is fine to pay conversion cost.
          throw std::invalid_argument(
              absl::StrFormat("Dict key mismatch; expected keys: %s; dict: %s.",
                              nb::cast<absl::string_view>(
                                  nb::repr(nb::cast(node.sorted_dict_keys))),
                              nb::cast<absl::string_view>(nb::repr(object))));
        }
        for (nb::handle key : keys) {
          agenda.push_back(dict[key]);
        }
        break;
      }

      case PyTreeKind::kNamedTuple: {
        if (!nb::isinstance<nb::tuple>(object) ||
            !nb::hasattr(object, "_fields")) {
          throw std::invalid_argument(
              absl::StrFormat("Expected named tuple, got %s.",
                              nb::cast<absl::string_view>(nb::repr(object))));
        }
        nb::tuple tuple = nb::borrow<nb::tuple>(object);
        if (tuple.size() != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "Named tuple arity mismatch: %d != %d; tuple: %s.", tuple.size(),
              node.arity, nb::cast<absl::string_view>(nb::repr(object))));
        }
        if (tuple.type().not_equal(node.node_data)) {
          throw std::invalid_argument(absl::StrFormat(
              "Named tuple type mismatch: expected type: %s, tuple: %s.",
              nb::cast<absl::string_view>(nb::repr(node.node_data)),
              nb::cast<absl::string_view>(nb::repr(object))));
        }
        for (nb::handle entry : tuple) {
          agenda.push_back(nb::borrow<nb::object>(entry));
        }
        break;
      }

      case PyTreeKind::kCustom: {
        auto* registration = registry_->Lookup(object.type());
        if (registration != node.custom) {
          throw std::invalid_argument(absl::StrFormat(
              "Custom node type mismatch: expected type: %s, value: %s.",
              nb::cast<absl::string_view>(nb::repr(node.custom->type)),
              nb::cast<absl::string_view>(nb::repr(object))));
        }
        auto [leaves, aux_data] = node.custom->ToIterable(object);
        if (node.node_data.not_equal(aux_data)) {
          throw std::invalid_argument(absl::StrFormat(
              "Mismatch custom node data: %s != %s; value: %s.",
              nb::cast<absl::string_view>(nb::repr(node.node_data)),
              nb::cast<absl::string_view>(nb::repr(aux_data)),
              nb::cast<absl::string_view>(nb::repr(object))));
        }
        int arity = 0;
        for (nb::handle entry : leaves) {
          ++arity;
          agenda.push_back(nb::borrow<nb::object>(entry));
        }
        if (arity != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "Custom type arity mismatch: %d != %d; value: %s.", arity,
              node.arity, nb::cast<absl::string_view>(nb::repr(object))));
        }
        break;
      }

      case PyTreeKind::kDataclass: {
        auto* registration = registry_->Lookup(object.type());
        if (registration != node.custom) {
          throw std::invalid_argument(absl::StrFormat(
              "Custom dataclasss node type mismatch: expected type: %s, value: "
              "%s.",
              nb::cast<absl::string_view>(nb::repr(node.custom->type)),
              nb::cast<absl::string_view>(nb::repr(std::move(object)))));
        }
        auto meta_size = node.custom->meta_fields.size();
        nb::object aux_data = nb::steal(PyTuple_New(meta_size));
        for (int meta_leaf = 0; meta_leaf < meta_size; ++meta_leaf) {
          PyTuple_SET_ITEM(
              aux_data.ptr(), meta_leaf,
              nb::getattr(object, node.custom->meta_fields[meta_leaf])
                  .release()
                  .ptr());
        }
        if (node.node_data.not_equal(aux_data)) {
          throw std::invalid_argument(absl::StrFormat(
              "Mismatch custom dataclass node data: %s != %s; value: %s.",
              nb::cast<absl::string_view>(nb::repr(node.node_data)),
              nb::cast<absl::string_view>(nb::repr(aux_data)),
              nb::cast<absl::string_view>(nb::repr(object))));
        }
        auto data_size = node.custom->data_fields.size();
        if (data_size != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "Custom type arity mismatch: %d != %d; value: %s.", data_size,
              node.arity, nb::cast<absl::string_view>(nb::repr(object))));
        }
        for (int leaf = 0; leaf < data_size; ++leaf) {
          agenda.push_back(nb::borrow<nb::object>(
              nb::getattr(object, node.custom->data_fields[leaf])));
        }
        break;
      }
    }
  }
  if (it != traversal_.rend() || leaf != -1) {
    throw std::invalid_argument(
        absl::StrFormat("Tree structures did not match: %s vs %s",
                        nb::cast<absl::string_view>(nb::repr(xs)), ToString()));
  }
  return leaves;
}

nb::object PyTreeDef::Walk(const nb::callable& f_node, nb::handle f_leaf,
                           nb::iterable leaves) const {
  std::vector<nb::object> agenda;
  auto it = leaves.begin();
  for (const Node& node : traversal_) {
    switch (node.kind) {
      case PyTreeKind::kLeaf: {
        if (it == leaves.end()) {
          throw std::invalid_argument("Too few leaves for PyTreeDef");
        }

        nb::object leaf = nb::borrow<nb::object>(*it);
        agenda.push_back(f_leaf.is_none() ? std::move(leaf)
                                          : f_leaf(std::move(leaf)));
        ++it;
        break;
      }

      case PyTreeKind::kNone:
      case PyTreeKind::kTuple:
      case PyTreeKind::kNamedTuple:
      case PyTreeKind::kList:
      case PyTreeKind::kDict:
      case PyTreeKind::kCustom:
      case PyTreeKind::kDataclass: {
        if (agenda.size() < node.arity) {
          throw std::logic_error("Too few elements for custom type.");
        }
        nb::object tuple = nb::steal(PyTuple_New(node.arity));
        for (int i = node.arity - 1; i >= 0; --i) {
          PyTuple_SET_ITEM(tuple.ptr(), i, agenda.back().release().ptr());
          agenda.pop_back();
        }
        nb::object node_data = node.node_data;
        if (node.kind == PyTreeKind::kDict) {
          // Convert to a nb::list for f_node invocation.
          node_data = nb::cast(node.sorted_dict_keys);
        }
        agenda.push_back(f_node(tuple, node_data ? node_data : nb::none()));
      }
    }
  }
  if (it != leaves.end()) {
    throw std::invalid_argument("Too many leaves for PyTreeDef");
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return std::move(agenda.back());
}

nb::object PyTreeDef::FromIterableTreeHelper(
    nb::handle xs,
    absl::InlinedVector<PyTreeDef::Node, 1>::const_reverse_iterator* it) const {
  if (*it == traversal_.rend()) {
    throw std::invalid_argument("Tree structures did not match.");
  }
  const Node& node = **it;
  ++*it;
  if (node.kind == PyTreeKind::kLeaf) {
    return nb::borrow<nb::object>(xs);
  }
  nb::iterable iterable = nb::borrow<nb::iterable>(xs);
  std::vector<nb::object> ys;
  ys.reserve(node.arity);
  for (nb::handle x : iterable) {
    ys.push_back(nb::borrow<nb::object>(x));
  }
  if (ys.size() != node.arity) {
    throw std::invalid_argument("Arity mismatch between trees");
  }
  for (int j = node.arity - 1; j >= 0; --j) {
    ys[j] = FromIterableTreeHelper(ys[j], it);
  }

  return MakeNode(node, absl::MakeSpan(ys));
}

nb::object PyTreeDef::FromIterableTree(nb::handle xs) const {
  auto it = traversal_.rbegin();
  nb::object out = FromIterableTreeHelper(xs, &it);
  if (it != traversal_.rend()) {
    throw std::invalid_argument("Tree structures did not match.");
  }
  return out;
}

nb_class_ptr<PyTreeDef> PyTreeDef::Compose(const PyTreeDef& inner) const {
  if (inner.registry_ != registry_) {
    throw std::invalid_argument(
        "PyTree registries of PyTreeDefs passed to Compose() must match.");
  }
  auto out = make_nb_class<PyTreeDef>(registry_ref_);
  out->traversal_.reserve(static_cast<size_t>(num_leaves()) *
                              inner.num_nodes() +
                          num_nodes() - num_leaves());
  for (const Node& n : traversal_) {
    if (n.kind == PyTreeKind::kLeaf) {
      absl::c_copy(inner.traversal_, std::back_inserter(out->traversal_));
    } else {
      out->traversal_.push_back(n);
    }
  }
  out->SetNumLeavesAndNumNodes();
  return out;
}

/*static*/ nb_class_ptr<PyTreeDef> PyTreeDef::Tuple(
    nb_class_ptr<PyTreeRegistry> registry, nb::list defs) {
  auto out = make_nb_class<PyTreeDef>(std::move(registry));
  int num_leaves = 0;
  for (nb::handle def_handle : defs) {
    const PyTreeDef* def = nb::cast<const PyTreeDef*>(def_handle);
    if (def->registry() != out->registry()) {
      throw std::invalid_argument(
          "PyTree registries of PyTreeDefs passed to Tuple() must match.");
    }
    absl::c_copy(def->traversal_, std::back_inserter(out->traversal_));
    num_leaves += def->num_leaves();
  }
  Node node;
  node.kind = PyTreeKind::kTuple;
  node.arity = defs.size();
  node.num_leaves = num_leaves;
  node.num_nodes = out->traversal_.size() + 1;
  out->traversal_.push_back(node);
  return out;
}

std::vector<nb_class_ptr<PyTreeDef>> PyTreeDef::Children() const {
  std::vector<nb_class_ptr<PyTreeDef>> children;
  if (traversal_.empty()) {
    return children;
  }
  Node const& root = traversal_.back();
  children.resize(root.arity);
  int pos = traversal_.size() - 1;
  for (int i = root.arity - 1; i >= 0; --i) {
    children[i] = make_nb_class<PyTreeDef>(registry_ref_);
    const Node& node = traversal_.at(pos - 1);
    if (pos < node.num_nodes) {
      throw std::logic_error("children() walked off start of array");
    }
    std::copy(traversal_.begin() + pos - node.num_nodes,
              traversal_.begin() + pos,
              std::back_inserter(children[i]->traversal_));
    pos -= node.num_nodes;
  }
  if (pos != 0) {
    throw std::logic_error("pos != 0 at end of PyTreeDef::Children");
  }
  return children;
}

std::string PyTreeDef::ToString() const {
  std::vector<std::string> agenda;
  for (const Node& node : traversal_) {
    if (agenda.size() < node.arity) {
      throw std::logic_error("Too few elements for container.");
    }

    std::string children =
        absl::StrJoin(agenda.end() - node.arity, agenda.end(), ", ");
    std::string representation;
    switch (node.kind) {
      case PyTreeKind::kLeaf:
        agenda.push_back("*");
        continue;
      case PyTreeKind::kNone:
        representation = "None";
        break;
      case PyTreeKind::kTuple:
        // Tuples with only one element must have a trailing comma.
        if (node.arity == 1) children += ",";
        representation = absl::StrCat("(", children, ")");
        break;
      case PyTreeKind::kList:
        representation = absl::StrCat("[", children, "]");
        break;
      case PyTreeKind::kDict: {
        if (node.sorted_dict_keys.size() != node.arity) {
          throw std::logic_error("Number of keys and entries does not match.");
        }
        representation = "{";
        std::string separator;
        auto child_iter = agenda.end() - node.arity;
        for (const nb::handle& key : node.sorted_dict_keys) {
          absl::StrAppendFormat(&representation, "%s%s: %s", separator,
                                nb::cast<absl::string_view>(nb::repr(key)),
                                *child_iter);
          child_iter++;
          separator = ", ";
        }
        representation += "}";
        break;
      }

      case PyTreeKind::kNamedTuple:
      case PyTreeKind::kCustom:
      case PyTreeKind::kDataclass: {
        std::string kind;
        std::string data;
        if (node.kind == PyTreeKind::kNamedTuple) {
          kind = "namedtuple";
          if (node.node_data) {
            // Node data for named tuples is the type.
            data = absl::StrFormat(
                "[%s]", nb::cast<absl::string_view>(
                            nb::str(nb::getattr(node.node_data, "__name__"))));
          }
        } else {
          kind = nb::cast<std::string>(
              nb::str(nb::getattr(node.custom->type, "__name__")));
          if (node.node_data) {
            data = absl::StrFormat(
                "[%s]", nb::cast<absl::string_view>(nb::str(node.node_data)));
          }
        }

        representation =
            absl::StrFormat("CustomNode(%s%s, [%s])", kind, data, children);
        break;
      }
    }
    agenda.erase(agenda.end() - node.arity, agenda.end());
    agenda.push_back(std::move(representation));
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return absl::StrCat("PyTreeDef(", agenda.back(), ")");
}

nb::object PyTreeDef::ToPickle() const {
  nb::list traversal;
  for (const auto& node : traversal_) {
    nb::object node_data = node.node_data;
    if (node.kind == PyTreeKind::kDict) {
      // Convert to a nb::list for pickling to avoid having to pickle a vector.
      // Pickle should be a rare operation so this conversion cost is hopefully
      // on non-critical path.
      node_data = nb::cast(node.sorted_dict_keys);
    }
    traversal.append(
        nb::make_tuple(static_cast<int>(node.kind), node.arity,
                       node_data ? node_data : nb::none(),
                       node.custom != nullptr ? node.custom->type : nb::none(),
                       node.num_leaves, node.num_nodes));
  }
  return nb::make_tuple(nb::cast(registry_ref_), traversal);
}

void PyTreeDef::FromPickle(nb::object pickle) {
  for (const auto& item : nb::cast<nb::list>(pickle)) {
    auto t = nb::cast<nb::tuple>(item);
    if (t.size() != 6) {
      throw xla::XlaRuntimeError("Malformed pickled PyTreeDef");
    }
    Node& node = traversal_.emplace_back();
    node.kind = static_cast<PyTreeKind>(nb::cast<int>(t[0]));
    node.arity = nb::cast<int>(t[1]);
    switch (node.kind) {
      case PyTreeKind::kNamedTuple:
        node.node_data = t[2];
        break;
      case PyTreeKind::kDict:
        node.sorted_dict_keys = nb::cast<std::vector<nb::object>>(t[2]);
        break;
      case PyTreeKind::kCustom:
      case PyTreeKind::kDataclass:
        node.node_data = t[2];
        break;
      default:
        if (!t[2].is_none()) {
          throw xla::XlaRuntimeError("Malformed pickled PyTreeDef");
        }
        break;
    }
    if (node.kind == PyTreeKind::kCustom ||
        node.kind == PyTreeKind::kDataclass) {
      node.custom = t[3].is_none() ? nullptr : registry()->Lookup(t[3]);
      if (node.custom == nullptr) {
        throw xla::XlaRuntimeError(
            absl::StrCat("Unknown custom type in pickled PyTreeDef: ",
                         nb::cast<absl::string_view>(nb::repr(t[3]))));
      }
    } else {
      if (!t[3].is_none()) {
        throw xla::XlaRuntimeError("Malformed pickled PyTreeDef");
      }
    }
    node.num_leaves = nb::cast<int>(t[4]);
    node.num_nodes = nb::cast<int>(t[5]);
  }
}

void PyTreeDef::SetNumLeavesAndNumNodes() {
  // num_leaves and num_nodes are fully determined by arity.
  std::vector<std::pair<int, int>> starts;
  int num_leaves = 0;
  for (int i = 0; i < traversal_.size(); ++i) {
    std::pair<int, int> start = {num_leaves, i};
    if (traversal_[i].kind == PyTreeKind::kLeaf) {
      num_leaves += 1;
    }
    if (traversal_[i].arity == 0) {
      starts.push_back(start);
    } else {
      starts.resize(starts.size() - (traversal_[i].arity - 1));
    }
    traversal_[i].num_leaves = num_leaves - starts.back().first;
    traversal_[i].num_nodes = i + 1 - starts.back().second;
  }
}

void PyTreeDef::SerializeTo(jax::PyTreeDefProto& result) const {
  absl::flat_hash_map<std::string, uint32_t> interned_strings;
  auto intern_str = [&](const std::string& key) {
    auto [it, added] =
        interned_strings.emplace(key, result.interned_strings_size());
    if (added) {
      result.add_interned_strings(key);
    }
    return it->second;
  };
  for (const auto& node : traversal_) {
    auto* node_data = result.add_nodes();
    node_data->set_arity(node.arity);
    switch (node.kind) {
      case PyTreeKind::kLeaf:
        node_data->set_type(jax::PyTreeNodeType::PY_TREE_KIND_LEAF);
        break;
      case PyTreeKind::kList:
        node_data->set_type(jax::PyTreeNodeType::PY_TREE_KIND_LIST);
        break;
      case PyTreeKind::kNone:
        node_data->set_type(jax::PyTreeNodeType::PY_TREE_KIND_NONE);
        break;
      case PyTreeKind::kTuple:
        node_data->set_type(jax::PyTreeNodeType::PY_TREE_KIND_TUPLE);
        break;
      case PyTreeKind::kDict:
        node_data->set_type(jax::PyTreeNodeType::PY_TREE_KIND_DICT);
        for (auto& key : node.sorted_dict_keys) {
          if (!nb::isinstance<nb::str>(key)) {
            throw std::invalid_argument(
                "Only string keys are supported in proto pytree "
                "serialization.");
          }
          node_data->mutable_dict_keys()->add_str_id(
              intern_str(nb::cast<std::string>(key)));
        }
        break;
      default:
        throw std::invalid_argument(
            "User-defined nodes are not supported when serializing pytrees as "
            "protocol buffers. You should either convert the user-defined "
            "nodes to another type or use pickle instead.");
        break;
    }
  }
}

nb_class_ptr<PyTreeDef> PyTreeDef::DeserializeFrom(
    nb_class_ptr<PyTreeRegistry> registry, const jax::PyTreeDefProto& input) {
  std::vector<nb::object> interned_strings;
  interned_strings.reserve(input.interned_strings().size());
  for (auto& s : input.interned_strings()) {
    interned_strings.push_back(nb::cast(s));
  }
  nb_class_ptr<PyTreeDef> result =
      make_nb_class<PyTreeDef>(std::move(registry));
  for (auto& node_proto : input.nodes()) {
    result->traversal_.emplace_back();
    auto& node = result->traversal_.back();
    node.arity = node_proto.arity();
    node.custom = nullptr;
    switch (node_proto.type()) {
      case jax::PyTreeNodeType::PY_TREE_KIND_LEAF:
        node.kind = PyTreeKind::kLeaf;
        break;
      case jax::PyTreeNodeType::PY_TREE_KIND_LIST:
        node.kind = PyTreeKind::kList;
        break;
      case jax::PyTreeNodeType::PY_TREE_KIND_NONE:
        node.kind = PyTreeKind::kNone;
        break;
      case jax::PyTreeNodeType::PY_TREE_KIND_TUPLE:
        node.kind = PyTreeKind::kTuple;
        break;
      case jax::PyTreeNodeType::PY_TREE_KIND_DICT:
        node.kind = PyTreeKind::kDict;
        for (uint32_t str_id : node_proto.dict_keys().str_id()) {
          if (str_id >= interned_strings.size()) {
            throw std::invalid_argument(
                "Malformed pytree proto (dict_key out of range).");
          }
          node.sorted_dict_keys.push_back(interned_strings.at(str_id));
        }
        break;
      default:
        throw std::invalid_argument(
            "Malformed pytree proto (invalid node type)");
        break;
    }
  }
  result->SetNumLeavesAndNumNodes();
  return result;
}

std::optional<std::pair<nb::object, nb::object>> PyTreeDef::GetNodeData()
    const {
  if (traversal_.empty()) {
    throw std::logic_error("empty PyTreeDef traversal.");
  }
  auto builtin_type = [](PyTypeObject* type_obj) {
    return nb::borrow<nb::object>(reinterpret_cast<PyObject*>(type_obj));
  };
  const auto& node = traversal_.back();
  switch (node.kind) {
    case PyTreeKind::kLeaf:
      return std::nullopt;
    case PyTreeKind::kNone:
      return std::make_pair(builtin_type(Py_TYPE(Py_None)), nb::none());
    case PyTreeKind::kTuple:
      return std::make_pair(builtin_type(&PyTuple_Type), nb::none());
    case PyTreeKind::kList:
      return std::make_pair(builtin_type(&PyList_Type), nb::none());
    case PyTreeKind::kDict:
      return std::make_pair(builtin_type(&PyDict_Type),
                            nb::cast(node.sorted_dict_keys));
    case PyTreeKind::kNamedTuple:
      return std::make_pair(node.node_data, nb::none());
    case PyTreeKind::kCustom:
    case PyTreeKind::kDataclass:
      return std::make_pair(node.custom->type, node.node_data);
  }
}

nb_class_ptr<PyTreeDef> PyTreeDef::MakeFromNodeDataAndChildren(
    nb_class_ptr<PyTreeRegistry> registry,
    std::optional<std::pair<nb::object, nb::object>> node_data,
    nb::iterable children) {
  nb_class_ptr<PyTreeDef> result =
      make_nb_class<PyTreeDef>(std::move(registry));
  int num_leaves = 0;
  int arity = 0;
  for (nb::handle pchild : children) {
    const PyTreeDef& child = nb::cast<const PyTreeDef&>(pchild);
    absl::c_copy(child.traversal_, std::back_inserter(result->traversal_));
    num_leaves += child.num_leaves();
    ++arity;
  }
  result->traversal_.emplace_back();
  auto& node = result->traversal_.back();
  node.arity = arity;
  node.custom = nullptr;
  node.num_leaves = num_leaves;
  node.num_nodes = result->traversal_.size();
  if (node_data == std::nullopt) {
    node.kind = PyTreeKind::kLeaf;
    ++node.num_leaves;
    return result;
  }
  int is_nt = PyObject_IsSubclass(node_data->first.ptr(),
                                  reinterpret_cast<PyObject*>(&PyTuple_Type));
  if (is_nt == -1) {
    throw nb::python_error();
  }
  if (is_nt != 0 && nb::hasattr(node_data->first, "_fields")) {
    node.kind = PyTreeKind::kNamedTuple;
    node.node_data = node_data->first;
    return result;
  }
  auto* registration = result->registry()->Lookup(node_data->first);
  if (registration == nullptr) {
    throw std::logic_error(absl::StrFormat(
        "Could not find type: %s.",
        nb::cast<absl::string_view>(nb::repr(node_data->first))));
  }
  node.kind = registration->kind;
  if (node.kind == PyTreeKind::kCustom || node.kind == PyTreeKind::kDataclass) {
    node.custom = registration;
    node.node_data = node_data->second;
  } else if (node.kind == PyTreeKind::kNamedTuple) {
    node.node_data = node_data->first;
  } else if (node.kind == PyTreeKind::kDict) {
    node.sorted_dict_keys =
        nb::cast<std::vector<nb::object>>(node_data->second);
  }
  return result;
}

int PyTreeDef::Node::tp_traverse(visitproc visit, void* arg) const {
  Py_VISIT(node_data.ptr());
  for (const auto& key : sorted_dict_keys) {
    Py_VISIT(key.ptr());
  }
  return 0;
}

/* static */ int PyTreeDef::tp_traverse(PyObject* self, visitproc visit,
                                        void* arg) {
  PyTreeDef* treedef = nb::inst_ptr<PyTreeDef>(self);
  Py_VISIT(Py_TYPE(self));
  Py_VISIT(treedef->registry_ref_.ptr());
  for (const auto& node : treedef->traversal_) {
    node.tp_traverse(visit, arg);
  }
  return 0;
}

/* static */ int PyTreeDef::tp_clear(PyObject* self) {
  PyTreeDef* treedef = nb::inst_ptr<PyTreeDef>(self);
  treedef->registry_ref_.reset();
  treedef->traversal_.clear();
  return 0;
}

/* static */ PyType_Slot PyTreeDef::slots_[] = {
    {Py_tp_traverse, (void*)PyTreeDef::tp_traverse},
    {Py_tp_clear, (void*)PyTreeDef::tp_clear},
    {0, nullptr},
};

void BuildPytreeSubmodule(nb::module_& m) {
  nb::module_ pytree = m.def_submodule("pytree", "Python tree library");
  pytree.attr("version") = nb::int_(3);

  nb::class_<PyTreeDef> treedef(pytree, "PyTreeDef",
                                nb::type_slots(PyTreeDef::slots_));

  nb::class_<PyTreeRegistry> registry(m, "PyTreeRegistry", nb::dynamic_attr(),
                                      nb::type_slots(PyTreeRegistry::slots_));

  registry.def(nb::init<bool, bool, bool, bool, bool>(),
               nb::arg("enable_none") = true, nb::arg("enable_tuple") = true,
               nb::arg("enable_namedtuple") = true,
               nb::arg("enable_list") = true, nb::arg("enable_dict") = true);
  registry.def(
      "flatten",
      [](nb_class_ptr<PyTreeRegistry> registry, nb::object x,
         std::optional<nb::callable> leaf_predicate) {
        nb::list leaves;
        nb_class_ptr<PyTreeDef> def =
            make_nb_class<PyTreeDef>(std::move(registry));
        def->Flatten(x, leaves, leaf_predicate);
        return nb::make_tuple(std::move(leaves), std::move(def));
      },
      nb::arg("tree").none(), nb::arg("leaf_predicate").none() = std::nullopt);
  registry.def("flatten_one_level", &PyTreeRegistry::FlattenOneLevel,
               nb::arg("tree").none());
  registry.def("flatten_one_level_with_keys",
               &PyTreeRegistry::FlattenOneLevelWithKeys,
               nb::arg("tree").none());
  registry.def(
      "flatten_with_path",
      [](nb_class_ptr<PyTreeRegistry> registry, nb::object x,
         std::optional<nb::callable> leaf_predicate) {
        nb::list leaves;
        nb_class_ptr<PyTreeDef> def =
            make_nb_class<PyTreeDef>(std::move(registry));
        def->FlattenWithPath(x, leaves, leaf_predicate);
        return nb::make_tuple(std::move(leaves), std::move(def));
      },
      nb::arg("tree").none(), nb::arg("leaf_predicate").none() = std::nullopt);
  registry.def("register_node", &PyTreeRegistry::Register,
               nb::arg("type").none(), nb::arg("to_iterable").none(),
               nb::arg("from_iterable").none(),
               nb::arg("to_iterable_with_keys").none() = std::nullopt);
  registry.def("register_dataclass_node", &PyTreeRegistry::RegisterDataclass);
  registry.def("__reduce__",
               [](nb::object self) { return self.attr("__name__"); });

  pytree.attr("_default_registry") = make_nb_class<PyTreeRegistry>(
      /*enable_none=*/true, /*enable_tuple=*/true, /*enable_namedtuple=*/true,
      /*enable_list=*/true, /*enable_dict*/ true);
  pytree.def("default_registry",
             [registry = nb::cast<nb_class_ptr<PyTreeRegistry>>(
                  pytree.attr("_default_registry"))]() { return registry; });

  pytree.attr("PyTreeRegistry") = m.attr("PyTreeRegistry");
  pytree.def("tuple", &PyTreeDef::Tuple);
  pytree.def("all_leaves", &PyTreeDef::AllLeaves);

  treedef.def("unflatten",
              static_cast<nb::object (PyTreeDef::*)(nb::iterable leaves) const>(
                  &PyTreeDef::Unflatten));
  treedef.def("flatten_up_to", &PyTreeDef::FlattenUpTo, nb::arg("tree").none());
  treedef.def("compose", &PyTreeDef::Compose);
  treedef.def(
      "walk", &PyTreeDef::Walk,
      "Walk pytree, calling f_node(node, node_data) at nodes, and f_leaf "
      "at leaves",
      nb::arg("f_node"), nb::arg("f_leaf"), nb::arg("leaves"));
  treedef.def("from_iterable_tree", &PyTreeDef::FromIterableTree);
  treedef.def("children", &PyTreeDef::Children);
  treedef.def_prop_ro("num_leaves", &PyTreeDef::num_leaves);
  treedef.def_prop_ro("num_nodes", &PyTreeDef::num_nodes);
  treedef.def("__repr__", &PyTreeDef::ToString);
  treedef.def("__eq__",
              [](const PyTreeDef& a, const PyTreeDef& b) { return a == b; });
  treedef.def("__ne__",
              [](const PyTreeDef& a, const PyTreeDef& b) { return a != b; });
  treedef.def("__hash__", [](const PyTreeDef& t) { return absl::HashOf(t); });
  treedef.def("serialize_using_proto", [](const PyTreeDef& a) {
    jax::PyTreeDefProto result;
    a.SerializeTo(result);
    std::string serialized = result.SerializeAsString();
    return nb::bytes(serialized.data(), serialized.size());
  });
  treedef.def_static(
      "deserialize_using_proto",
      [](nb_class_ptr<PyTreeRegistry> registry, nb::bytes data) {
        jax::PyTreeDefProto input;
        absl::string_view serialized(data.c_str(), data.size());
        if (serialized.size() > std::numeric_limits<int>::max()) {
          throw xla::XlaRuntimeError(
              "Pytree serialization too large to deserialize.");
        }
        if (!input.ParseFromArray(serialized.data(), serialized.size())) {
          throw xla::XlaRuntimeError("Could not deserialize PyTreeDefProto.");
        }
        return PyTreeDef::DeserializeFrom(std::move(registry), input);
      },
      nb::arg("registry"), nb::arg("data"));
  treedef.def("node_data", &PyTreeDef::GetNodeData,
              "Returns None if a leaf-pytree, else (type, node_data)");
  treedef.def_static(
      "make_from_node_data_and_children",
      &PyTreeDef::MakeFromNodeDataAndChildren, nb::arg("registry"),
      nb::arg("node_data").none(), nb::arg("children"),
      "Reconstructs a pytree from `node_data()` and `children()`.");
  treedef.def("__getstate__", &PyTreeDef::ToPickle);
  treedef.def("__setstate__", [](PyTreeDef& t, nb::object o) {
    nb::tuple pickle = nb::cast<nb::tuple>(o);
    if (pickle.size() != 2) {
      throw xla::XlaRuntimeError(
          "Malformed pickled PyTreeDef, expected 2-tuple");
    }
    auto registry = nb::cast<nb_class_ptr<PyTreeRegistry>>(pickle[0]);
    new (&t) PyTreeDef(registry);
    t.FromPickle(pickle[1]);
  });

  nb::class_<SequenceKey> sequence_key(pytree, "SequenceKey");
  sequence_key.def(nb::init<int>(), nb::arg("idx"));
  sequence_key.def("__str__", &SequenceKey::ToString);
  sequence_key.def("__repr__", &SequenceKey::ToReprString);
  sequence_key.def("__eq__", &SequenceKey::Equals);
  sequence_key.def("__hash__", [](const SequenceKey& key) {
    return key.idx() + kSequenceKeyHashSalt;
  });
  sequence_key.def_prop_ro("idx", &SequenceKey::idx);
  sequence_key.def_prop_ro_static("__match_args__", &SequenceKey::MatchArgs);
  sequence_key.def("__getstate__",
                   [](SequenceKey& key) { return nb::make_tuple(key.idx()); });
  sequence_key.def("__setstate__",
                   [](SequenceKey& key, const nb::tuple& state) {
                     if (state.size() != 1) {
                       throw xla::XlaRuntimeError(
                           "Malformed pickled SequenceKey, expected 1-tuple");
                     }
                     new (&key) SequenceKey(nb::cast<int>(state[0]));
                   });

  nb::class_<DictKey> dict_key(pytree, "DictKey",
                               nb::type_slots(DictKey::slots_));
  dict_key.def(nb::init<nb::object>(), nb::arg("key"));
  dict_key.def("__str__", &DictKey::ToString);
  dict_key.def("__repr__", &DictKey::ToReprString);
  dict_key.def("__eq__", &DictKey::Equals);
  dict_key.def("__hash__",
               [](const DictKey& key) { return nanobind::hash(key.key()); });
  dict_key.def_prop_ro("key", &DictKey::key);
  dict_key.def_prop_ro_static("__match_args__", &DictKey::MatchArgs);
  dict_key.def("__getstate__",
               [](DictKey& key) { return nb::make_tuple(key.key()); });
  dict_key.def("__setstate__", [](DictKey& key, const nb::tuple& state) {
    if (state.size() != 1) {
      throw xla::XlaRuntimeError("Malformed pickled DictKey, expected 1-tuple");
    }
    new (&key) DictKey(nb::cast<nb::object>(state[0]));
  });

  nb::class_<GetAttrKey> get_attr_key(pytree, "GetAttrKey");
  get_attr_key.def(nb::init<nb::str>(), nb::arg("name"));
  get_attr_key.def("__str__", &GetAttrKey::ToString);
  get_attr_key.def("__repr__", &GetAttrKey::ToReprString);
  get_attr_key.def("__eq__", &GetAttrKey::Equals);
  get_attr_key.def("__hash__",
                   [](const GetAttrKey& key) { return nb::hash(key.name()); });
  get_attr_key.def_prop_ro("name", &GetAttrKey::name);
  get_attr_key.def_prop_ro_static("__match_args__", &GetAttrKey::MatchArgs);
  get_attr_key.def("__getstate__",
                   [](GetAttrKey& key) { return nb::make_tuple(key.name()); });
  get_attr_key.def("__setstate__", [](GetAttrKey& key, const nb::tuple& state) {
    if (state.size() != 1) {
      throw xla::XlaRuntimeError(
          "Malformed pickled GetAttrKey, expected 1-tuple");
    }
    new (&key) GetAttrKey(nb::str(state[0]));
  });

  nb::class_<FlattenedIndexKey> flattened_index_key(pytree,
                                                    "FlattenedIndexKey");
  flattened_index_key.def(nb::init<int>(), nb::arg("key"));
  flattened_index_key.def("__str__", &FlattenedIndexKey::ToString);
  flattened_index_key.def("__repr__", &FlattenedIndexKey::ToReprString);
  flattened_index_key.def("__eq__", &FlattenedIndexKey::Equals);
  flattened_index_key.def("__hash__", [](const FlattenedIndexKey& key) {
    return key.key() + kFlattenedIndexKeyHashSalt;
  });
  flattened_index_key.def_prop_ro("key", &FlattenedIndexKey::key);
  flattened_index_key.def_prop_ro_static("__match_args__",
                                         &FlattenedIndexKey::MatchArgs);
  flattened_index_key.def("__getstate__", [](FlattenedIndexKey& key) {
    return nb::make_tuple(key.key());
  });
  flattened_index_key.def(
      "__setstate__", [](FlattenedIndexKey& key, const nb::tuple& state) {
        if (state.size() != 1) {
          throw xla::XlaRuntimeError(
              "Malformed pickled FlattenedIndexKey, expected 1-tuple");
        }
        new (&key) FlattenedIndexKey(nb::cast<int>(state[0]));
      });
}

}  // namespace xla
