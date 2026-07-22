/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_UTIL_INTERVAL_TREE_H_
#define XLA_UTIL_INTERVAL_TREE_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <list>
#include <string>
#include <type_traits>

#include "absl/strings/str_cat.h"

namespace xla {

template <typename T, typename = void>
struct has_to_string : std::false_type {};

template <typename T>
struct has_to_string<T, std::void_t<decltype(std::declval<T>().ToString())>>
    : std::is_convertible<decltype(std::declval<T>().ToString()), std::string> {
};

template <typename ValueType>
struct IntervalTreeNode {
  static_assert(has_to_string<ValueType>::value,
                "ValueType must have a ToString() member function returning "
                "std::string (or convertible to it)");
  // Alloc time.
  int64_t start;
  // Free time.
  int64_t end;
  // Maximum free time of all nodes in the subtree where this node is the root.
  int64_t subtree_end;
  // Allocated chunk for the buffer.
  ValueType value;
  // Left child.
  IntervalTreeNode* left;
  // Right child.
  IntervalTreeNode* right;
  // parent
  IntervalTreeNode* parent;

  std::string ToString() const {
    return absl::StrCat("start: ", start, " end: ", end,
                        " value: ", value.ToString());
  }
};

template <typename ValueType>
class IntervalTree {
  static_assert(has_to_string<ValueType>::value,
                "ValueType must have a ToString() member function returning "
                "std::string (or convertible to it)");

 public:
  void Add(int64_t start, int64_t end, const ValueType& value) {
    node_storage_.emplace_back(IntervalTreeNode<ValueType>{
        start, end, end, value,
        /*left=*/nullptr, /*right=*/nullptr, /*parent=*/nullptr});
    if (root_ == nullptr) {
      root_ = &node_storage_.back();
      return;
    }

    IntervalTreeNode<ValueType>* parent = root_;
    while (true) {
      parent->subtree_end = std::max(parent->subtree_end, end);
      if (parent->start > start) {
        if (parent->left == nullptr) {
          parent->left = &node_storage_.back();
          node_storage_.back().parent = parent;
          return;
        }
        parent = parent->left;
      } else {
        if (parent->right == nullptr) {
          parent->right = &node_storage_.back();
          node_storage_.back().parent = parent;
          return;
        }
        parent = parent->right;
      }
    }
  }

  bool Remove(int64_t start, int64_t end, const ValueType& value) {
    IntervalTreeNode<ValueType>* to_delete = root_;
    while (to_delete != nullptr) {
      if (to_delete->start == start && to_delete->end == end &&
          to_delete->value.offset == value.offset) {
        break;
      }
      if (start < to_delete->start) {
        to_delete = to_delete->left;
      } else {
        to_delete = to_delete->right;
      }
    }
    if (to_delete == nullptr) {
      return false;
    }

    std::function<void(IntervalTreeNode<ValueType>*)> fix_up =
        [&](IntervalTreeNode<ValueType>* node) {
          if (node == nullptr) {
            return;
          }
          node->subtree_end = node->end;
          if (node->left) {
            node->subtree_end =
                std::max(node->subtree_end, node->left->subtree_end);
          }
          if (node->right) {
            node->subtree_end =
                std::max(node->subtree_end, node->right->subtree_end);
          }
          fix_up(node->parent);
        };

    if (to_delete->right == nullptr) {
      if (root_ == to_delete) {
        root_ = to_delete->left;
        return true;
      }

      if (to_delete == to_delete->parent->left) {
        to_delete->parent->left = to_delete->left;
      }
      if (to_delete == to_delete->parent->right) {
        to_delete->parent->right = to_delete->left;
      }
      if (to_delete->left) {
        to_delete->left->parent = to_delete->parent;
      }
      fix_up(to_delete);
    } else {
      IntervalTreeNode<ValueType>* to_promote = to_delete->right;
      while (to_promote->left != nullptr) {
        to_promote = to_promote->left;
      }

      to_delete->start = to_promote->start;
      to_delete->end = to_promote->end;
      to_delete->subtree_end = to_promote->subtree_end;
      to_delete->value = to_promote->value;
      auto to_promote_parent = to_promote->parent;
      if (to_promote_parent->left == to_promote) {
        to_promote_parent->left = to_promote->right;
      } else {
        to_promote_parent->right = to_promote->right;
      }
      if (to_promote->right) {
        to_promote->right->parent = to_promote_parent;
      }
      fix_up(to_promote_parent);
    }
    return true;
  }

 protected:
  IntervalTreeNode<ValueType>* root_ = nullptr;
  std::list<IntervalTreeNode<ValueType>> node_storage_;
};

}  // namespace xla

#endif  // XLA_UTIL_INTERVAL_TREE_H_
