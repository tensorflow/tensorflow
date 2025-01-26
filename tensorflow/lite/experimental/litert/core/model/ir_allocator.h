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

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
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

  // Transfers the ownership of given allocator to this one.
  void Transfer(IrAllocator&& other) {
    storage_.splice(storage_.cend(), other.storage_);
    refs_->insert(refs_->end(), other.refs_->cbegin(), other.refs_->cend());
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
  Storage storage_;
  std::unique_ptr<Refs> refs_;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_IR_ALLOCATOR_H_
