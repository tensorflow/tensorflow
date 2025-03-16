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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_IR_ALLOCATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_IR_ALLOCATOR_H_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"

namespace litert::internal {

// A list of IR objects scoped to the same block (subgraph) that provides
// pointer stability. Facilitates management of memory and c-like access
// to elements.
template <class Ir>
class IrAllocator {
 private:
  using Storage = std::list<Ir>;
  using Refs = std::vector<Ir*>;

 public:
  // Emplace a new element onto the list.
  template <class... Args>
  Ir& EmplaceBack(Args&&... args) {
    auto& emp = storage_.emplace_back(std::forward<Args>(args)...);
    refs_->push_back(&emp);
    return emp;
  }

  // Get the array of (stable) pointers to underlying elements. Suitable
  // for passing through c-like interface. Consituent pointers are always
  // guarateed to be stable (unless explicitly erased). The array of pointers
  // itself is guaranteed to be stable so long as no length-changing operations
  // occur, moving this class does not invalidate pointers or array.
  absl::Span<Ir*> Elements() const {
    return absl::MakeSpan(refs_->data(), refs_->size());
  }

  // Remove elements from the allocator if they match the predicate.
  // Returns the number of elements removed.
  size_t RemoveIf(std::function<bool(const Ir& ir)> pred) {
    auto ref_it = refs_->begin();
    for (auto it = storage_.begin(); it != storage_.end();) {
      if (!pred(*it)) {
        *ref_it = &*it;
        ++ref_it;
        ++it;
        continue;
      }
      it = storage_.erase(it);
    }
    const size_t removed = refs_->end() - ref_it;
    refs_->resize(refs_->size() - removed);
    return removed;
  }

  // Cuts all but the first `size` elements from storage. Does nothing if `size`
  // is greater or equal to current size.
  void ResizeDown(size_t size) {
    if (size >= Size()) {
      return;
    }
    storage_.resize(size);
    refs_->resize(size);
  }

  // Transfers the ownership of given allocator to this one. If `indices` is
  // provided, only the objects at the given indices are transferred.
  void TransferFrom(IrAllocator& other,
                    std::optional<std::vector<size_t>> indices = std::nullopt) {
    if (!indices) {
      storage_.splice(storage_.cend(), other.storage_);
      refs_->insert(refs_->end(), other.refs_->cbegin(), other.refs_->cend());
      other.ResetRefs();
      return;
    }

    auto& inds = *indices;
    std::sort(inds.begin(), inds.end());
    std::vector<typename Storage::iterator> its;
    auto i = 0;
    auto it = other.storage_.begin();
    for (auto ind : inds) {
      std::advance(it, ind - i);
      i = ind;
      its.push_back(it);
    }
    for (auto it : its) {
      storage_.splice(storage_.cend(), other.storage_, it);
    }

    ResetRefs();
    other.ResetRefs();
  }

  // Override for rvalues.
  void TransferFrom(IrAllocator&& other) { TransferFrom(other, std::nullopt); }

  // Transfers the object at the given index to the back of the given allocator.
  void TransferTo(IrAllocator& other,
                  std::optional<std::vector<size_t>> indices = std::nullopt) {
    other.TransferFrom(*this, std::move(indices));
  }

  // Number of elements stored by this allocator.
  size_t Size() const { return storage_.size(); }

  IrAllocator() { refs_ = std::make_unique<Refs>(); }

  // IR is generally semantically movable (without reference invalidation)
  // but not copyable. IrAllocators reflect that, note moving lists
  // does not invalidate references.
  IrAllocator(const IrAllocator& other) = delete;
  IrAllocator& operator=(const IrAllocator& other) = delete;
  IrAllocator(IrAllocator&& other) = default;
  IrAllocator& operator=(IrAllocator&& other) = default;

 private:
  void ResetRefs() {
    refs_->resize(storage_.size());
    auto it = storage_.begin();
    for (auto i = 0; i < storage_.size(); ++i, ++it) {
      refs_->at(i) = &*it;
    }
  }

  Storage storage_;
  std::unique_ptr<Refs> refs_;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_IR_ALLOCATOR_H_
