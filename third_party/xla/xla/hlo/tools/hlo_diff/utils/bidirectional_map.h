/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_UTILS_BIDIRECTIONAL_MAP_H_
#define XLA_HLO_TOOLS_HLO_DIFF_UTILS_BIDIRECTIONAL_MAP_H_

#include <cstddef>
#include <iterator>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"

// A absl-based bidirectional map with optional associated properties.
// This class encapsulates the map implementation and provides methods for
// efficient lookups from either left-to-right or right-to-left.
template <typename LeftT, typename RightT, typename PropsT>
class BidirectionalMap {
 private:
  struct ForwardMapping {
    RightT node;
    std::optional<PropsT> props;
  };

  using underlying_map_t = absl::flat_hash_map<LeftT, ForwardMapping>;

 public:
  class const_iterator {
   public:
    // Public iterator traits
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;

    // A proxy object returned by the iterator to provide direct access to
    // the left and right elements of the mapping, mimicking std::map's iterator
    // behavior.
    using reference = std::pair<const LeftT, const RightT>;
    using pointer = std::pair<const LeftT, const RightT>;

    const_iterator() = default;

    pointer operator->() const {
      return std::make_pair(it_->first, it_->second.node);
    }
    reference operator*() const {
      return std::make_pair(it_->first, it_->second.node);
    }

    const_iterator& operator++() {
      ++it_;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const const_iterator& a, const const_iterator& b) {
      return a.it_ == b.it_;
    }
    friend bool operator!=(const const_iterator& a, const const_iterator& b) {
      return a.it_ != b.it_;
    }

   private:
    friend class BidirectionalMap;
    explicit const_iterator(typename underlying_map_t::const_iterator it)
        : it_(it) {}

    typename underlying_map_t::const_iterator it_;
  };

  // Inserts a new mapping between the left and right elements with associated
  // properties. Returns true if the mapping was inserted, false otherwise.
  template <typename L, typename R, typename P>
  bool Insert(L&& left_element, R&& right_element, P&& props) {
    if (ContainsLeft(left_element) || ContainsRight(right_element)) {
      return false;
    }
    auto [it, success] = left_to_right_.emplace(
        std::forward<L>(left_element),
        ForwardMapping{std::forward<R>(right_element), std::forward<P>(props)});

    // Use the newly inserted values to create the reverse mapping
    if (success) {
      right_to_left_.emplace(it->second.node, it->first);
    }
    return success;
  }

  // Similar as above for inserting without properties.
  template <typename L, typename R>
  bool Insert(L&& left_element, R&& right_element) {
    if (ContainsLeft(left_element) || ContainsRight(right_element)) {
      return false;
    }
    auto [it, success] = left_to_right_.emplace(
        std::forward<L>(left_element),
        ForwardMapping{std::forward<R>(right_element), std::nullopt});

    if (success) {
      right_to_left_.emplace(it->second.node, it->first);
    }
    return success;
  }

  // Deletes a mapping by its left element. Returns true if the element was
  // found and deleted, false otherwise.
  bool EraseByLeft(const LeftT& left_element) {
    auto it = left_to_right_.find(left_element);
    if (it == left_to_right_.end()) {
      return false;
    }

    right_to_left_.erase(it->second.node);
    left_to_right_.erase(it);
    return true;
  }

  // Deletes a mapping by its right element. Returns true if the element was
  // found and deleted, false otherwise.
  bool EraseByRight(const RightT& right_element) {
    auto it = right_to_left_.find(right_element);
    if (it == right_to_left_.end()) {
      return false;
    }

    left_to_right_.erase(it->second);
    right_to_left_.erase(it);
    return true;
  }

  // Returns the number of mappings in the map.
  size_t size() const { return left_to_right_.size(); }

  // Returns true if the left element is present in the map.
  bool ContainsLeft(const LeftT& left_element) const {
    return left_to_right_.contains(left_element);
  }

  // Returns true if the right element is present in the map.
  bool ContainsRight(const RightT& right_element) const {
    return right_to_left_.contains(right_element);
  }

  // Returns the right element associated with the left element, if present.
  std::optional<RightT> GetRight(const LeftT& left_element) const {
    if (auto it = left_to_right_.find(left_element);
        it != left_to_right_.end()) {
      return it->second.node;
    }
    return std::nullopt;
  }

  // Returns the left element associated with the right element, if present.
  std::optional<LeftT> GetLeft(const RightT& right_element) const {
    if (auto it = right_to_left_.find(right_element);
        it != right_to_left_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  // Returns the properties associated with the left element, if present.
  std::optional<PropsT> GetPropsByLeft(const LeftT& left_element) const {
    auto it = left_to_right_.find(left_element);
    if (it == left_to_right_.end() || !it->second.props.has_value()) {
      return std::nullopt;
    }

    return it->second.props;
  }

  // Returns the properties associated with the right element, if present.
  std::optional<PropsT> GetPropsByRight(const RightT& right_element) const {
    std::optional<LeftT> left = GetLeft(right_element);
    return left.has_value() ? GetPropsByLeft(left.value()) : std::nullopt;
  }

  // Sets the properties associated with the left element. Returns true if the
  // left element is present in the map, false otherwise.
  bool SetPropsByLeft(const LeftT& left_element, PropsT props) {
    auto it = left_to_right_.find(left_element);
    if (it == left_to_right_.end()) {
      return false;
    }

    it->second.props = props;
    return true;
  }

  // --- Iteration Methods ---
  const_iterator begin() const {
    return const_iterator(left_to_right_.begin());
  }
  const_iterator end() const { return const_iterator(left_to_right_.end()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

 private:
  absl::flat_hash_map<LeftT, ForwardMapping> left_to_right_;
  absl::flat_hash_map<RightT, LeftT> right_to_left_;
};

#endif  // XLA_HLO_TOOLS_HLO_DIFF_UTILS_BIDIRECTIONAL_MAP_H_
