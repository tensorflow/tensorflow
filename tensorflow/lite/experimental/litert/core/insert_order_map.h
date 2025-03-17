// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_INSERT_ORDER_MAP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_INSERT_ORDER_MAP_H_

#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace litert::internal {

// A map implementation that iterates in the same order as initial insertion.
template <class Key, class Val>
class InsertOrderMap {
 public:
  using Pair = std::pair<Key, Val>;
  using Values = std::vector<Pair>;
  using ValRef = std::reference_wrapper<Pair>;
  using Map = absl::flat_hash_map<Key, size_t>;
  using Iterator = typename Values::iterator;

  InsertOrderMap() = default;

  std::optional<ValRef> Find(const Key& key) {
    if (auto it = map_.find(key); it != map_.end()) {
      const auto ind = it->second;
      return values_[ind];
    }
    return {};
  }

  bool Contains(const Key& key) const { return map_.find(key) != map_.end(); }

  void InsertOrAssign(const Key& key, const Val& val) {
    if (auto it = map_.find(key); it != map_.end()) {
      const auto ind = it->second;
      values_[ind].second = val;
    } else {
      values_.push_back({key, val});
      map_.insert({key, values_.size() - 1});
    }
  }

  size_t Size() const { return values_.size(); }

  void Clear() {
    values_.clear();
    map_.clear();
  }

  Iterator Begin() { return values_.begin(); }

  Iterator End() { return values_.end(); }

 private:
  Values values_;
  Map map_;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_INSERT_ORDER_MAP_H_
