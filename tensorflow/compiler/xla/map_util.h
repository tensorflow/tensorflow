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

#ifndef TENSORFLOW_COMPILER_XLA_MAP_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_MAP_UTIL_H_

#include <functional>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// FindOrDie returns a const reference to the value associated with
// the given key if it exists. Crashes otherwise.
//
// This is intended as a replacement for operator[] as an rvalue (for reading)
// when the key is guaranteed to exist.
template <class Collection>
const typename Collection::value_type::second_type& FindOrDie(
    const Collection& collection,
    const typename Collection::value_type::first_type& key) {
  typename Collection::const_iterator it = collection.find(key);
  CHECK(it != collection.end()) << "Map key not found: " << key;
  return it->second;
}

// Same as above, but returns a non-const reference.
template <class Collection>
typename Collection::value_type::second_type& FindOrDie(
    Collection& collection,  // NOLINT
    const typename Collection::value_type::first_type& key) {
  typename Collection::iterator it = collection.find(key);
  CHECK(it != collection.end()) << "Map key not found: " << key;
  return it->second;
}

// Like FindOrDie but returns an error instead of dying if `key` is not in
// `container`.
template <class Collection>
StatusOr<
    std::reference_wrapper<const typename Collection::value_type::second_type>>
MaybeFind(const Collection& collection,
          const typename Collection::value_type::first_type& key) {
  typename Collection::const_iterator it = collection.find(key);
  if (it == collection.end()) {
    std::ostringstream os;
    os << key;
    return NotFound("key not found: %s", os.str());
  }
  return {it->second};
}

// Returns a const reference to the value associated with the given key if it
// exists, otherwise returns a const reference to the provided default value.
//
// WARNING: If a temporary object is passed as the default "value,"
// this function will return a reference to that temporary object,
// which will be destroyed at the end of the statement. A common
// example: if you have a map with string values, and you pass a char*
// as the default "value," either use the returned value immediately
// or store it in a string (not string&).
template <class Collection>
const typename Collection::value_type::second_type& FindOrDefault(
    const Collection& collection,
    const typename Collection::value_type::first_type& key,
    const typename Collection::value_type::second_type& value) {
  auto it = collection.find(key);
  if (it != collection.end()) return it->second;
  return value;
}

// Inserts the key-value pair into the collection. Dies if key was already
// present.
template <class Collection, class Key, class Value>
void InsertOrDie(Collection* const collection, Key&& key, Value&& value) {
  auto p = collection->insert(
      std::make_pair(std::forward<Key>(key), std::forward<Value>(value)));
  CHECK(p.second) << "duplicate key: " << key;
}

// Returns true if and only if the given collection contains the given key.
template <class Collection, class Key>
bool ContainsKey(const Collection& collection, const Key& key) {
  return collection.find(key) != collection.end();
}

// Returns a function that returns whether the map contains the given key.
template <class Key, class Value>
auto IsKeyIn(const absl::flat_hash_map<Key, Value>& map) {
  return [&](const Key& key) { return map.contains(key); };
}

// Returns a function that returns whether the set contains the given value.
template <class T>
auto IsValueIn(const absl::flat_hash_set<T>& set) {
  return [&](const T& value) { return set.contains(value); };
}

// Inserts `value` into `set`. Dies if it was already present.
template <class Set, class Value>
void InsertOrDie(Set* const set, Value&& value) {
  CHECK(set->insert(std::forward<Value>(value)).second)
      << "duplicate value: " << value;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MAP_UTIL_H_
