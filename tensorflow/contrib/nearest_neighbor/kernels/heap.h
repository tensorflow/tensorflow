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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_NEAREST_NEIGHBOR_KERNELS_HEAP_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_NEAREST_NEIGHBOR_KERNELS_HEAP_H_

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace tensorflow {
namespace nearest_neighbor {

// A simple binary heap. We use our own implementation because multiprobe for
// the cross-polytope hash interacts with the heap in a way so that about half
// of the insertion operations are guaranteed to be on top of the heap. We make
// use of this fact in the AugmentedHeap below.

// HeapBase is a base class for both the SimpleHeap and AugmentedHeap below.
template <typename KeyType, typename DataType>
class HeapBase {
 public:
  class Item {
   public:
    KeyType key;
    DataType data;

    Item() {}
    Item(const KeyType& k, const DataType& d) : key(k), data(d) {}

    bool operator<(const Item& i2) const { return key < i2.key; }
  };

  void ExtractMin(KeyType* key, DataType* data) {
    *key = v_[0].key;
    *data = v_[0].data;
    num_elements_ -= 1;
    v_[0] = v_[num_elements_];
    HeapDown(0);
  }

  bool IsEmpty() { return num_elements_ == 0; }

  // This method adds an element at the end of the internal array without
  // "heapifying" the array afterwards. This is useful for setting up a heap
  // where a single call to heapify at the end of the inital insertion
  // operations suffices.
  void InsertUnsorted(const KeyType& key, const DataType& data) {
    if (v_.size() == static_cast<size_t>(num_elements_)) {
      v_.push_back(Item(key, data));
    } else {
      v_[num_elements_].key = key;
      v_[num_elements_].data = data;
    }
    num_elements_ += 1;
  }

  void Insert(const KeyType& key, const DataType& data) {
    if (v_.size() == static_cast<size_t>(num_elements_)) {
      v_.push_back(Item(key, data));
    } else {
      v_[num_elements_].key = key;
      v_[num_elements_].data = data;
    }
    num_elements_ += 1;
    HeapUp(num_elements_ - 1);
  }

  void Heapify() {
    int_fast32_t rightmost = parent(num_elements_ - 1);
    for (int_fast32_t cur_loc = rightmost; cur_loc >= 0; --cur_loc) {
      HeapDown(cur_loc);
    }
  }

  void Reset() { num_elements_ = 0; }

  void Resize(size_t new_size) { v_.resize(new_size); }

 protected:
  int_fast32_t lchild(int_fast32_t x) { return 2 * x + 1; }

  int_fast32_t rchild(int_fast32_t x) { return 2 * x + 2; }

  int_fast32_t parent(int_fast32_t x) { return (x - 1) / 2; }

  void SwapEntries(int_fast32_t a, int_fast32_t b) {
    Item tmp = v_[a];
    v_[a] = v_[b];
    v_[b] = tmp;
  }

  void HeapUp(int_fast32_t cur_loc) {
    int_fast32_t p = parent(cur_loc);
    while (cur_loc > 0 && v_[p].key > v_[cur_loc].key) {
      SwapEntries(p, cur_loc);
      cur_loc = p;
      p = parent(cur_loc);
    }
  }

  void HeapDown(int_fast32_t cur_loc) {
    while (true) {
      int_fast32_t lc = lchild(cur_loc);
      int_fast32_t rc = rchild(cur_loc);
      if (lc >= num_elements_) {
        return;
      }

      if (v_[cur_loc].key <= v_[lc].key) {
        if (rc >= num_elements_ || v_[cur_loc].key <= v_[rc].key) {
          return;
        } else {
          SwapEntries(cur_loc, rc);
          cur_loc = rc;
        }
      } else {
        if (rc >= num_elements_ || v_[lc].key <= v_[rc].key) {
          SwapEntries(cur_loc, lc);
          cur_loc = lc;
        } else {
          SwapEntries(cur_loc, rc);
          cur_loc = rc;
        }
      }
    }
  }

  std::vector<Item> v_;
  int_fast32_t num_elements_ = 0;
};

// A "simple" binary heap.
template <typename KeyType, typename DataType>
class SimpleHeap : public HeapBase<KeyType, DataType> {
 public:
  void ReplaceTop(const KeyType& key, const DataType& data) {
    this->v_[0].key = key;
    this->v_[0].data = data;
    this->HeapDown(0);
  }

  KeyType MinKey() { return this->v_[0].key; }

  std::vector<typename HeapBase<KeyType, DataType>::Item>& GetData() {
    return this->v_;
  }
};

// An "augmented" heap that can hold an extra element that is guaranteed to
// be at the top of the heap. This is useful if a significant fraction of the
// insertion operations are guaranteed insertions at the top. However, the heap
// only stores at most one such special top element, i.e., the heap assumes
// that extract_min() is called at least once between successive calls to
// insert_guaranteed_top().
template <typename KeyType, typename DataType>
class AugmentedHeap : public HeapBase<KeyType, DataType> {
 public:
  void ExtractMin(KeyType* key, DataType* data) {
    if (has_guaranteed_top_) {
      has_guaranteed_top_ = false;
      *key = guaranteed_top_.key;
      *data = guaranteed_top_.data;
    } else {
      *key = this->v_[0].key;
      *data = this->v_[0].data;
      this->num_elements_ -= 1;
      this->v_[0] = this->v_[this->num_elements_];
      this->HeapDown(0);
    }
  }

  bool IsEmpty() { return this->num_elements_ == 0 && !has_guaranteed_top_; }

  void InsertGuaranteedTop(const KeyType& key, const DataType& data) {
    assert(!has_guaranteed_top_);
    has_guaranteed_top_ = true;
    guaranteed_top_.key = key;
    guaranteed_top_.data = data;
  }

  void Reset() {
    this->num_elements_ = 0;
    has_guaranteed_top_ = false;
  }

 protected:
  typename HeapBase<KeyType, DataType>::Item guaranteed_top_;
  bool has_guaranteed_top_ = false;
};

}  // namespace nearest_neighbor
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_NEAREST_NEIGHBOR_KERNELS_HEAP_H_
