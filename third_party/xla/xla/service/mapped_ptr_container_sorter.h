/* Copyright 2022 The OpenXLA Authors.

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

// Below, we specify an example usage, in which clone is sorted according to
// original, using map_fn to map from pointers in original to pointers in clone.
//
//   std::vector<std::unique_ptr<HloInstruction*>> original = ...;
//   std::vector<std::unique_ptr<HloInstruction*>> clone = ...;
//   HloCloneContext* ctx = ...;
//   using Sorter = MappedPtrContainerSorter<HloInstruction>;
//   Sorter::MappedPtrFn map_fn = [ctx](const HloInstruction* i) {
//       return ctx->FindInstruction(i);
//     };
//
//   auto status = Sorter::Sort(map_fn, Sorter::IndexAfterMappedElementsFn(),
//                              original, clone);

#ifndef XLA_SERVICE_MAPPED_PTR_CONTAINER_SORTER_H_
#define XLA_SERVICE_MAPPED_PTR_CONTAINER_SORTER_H_

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {

// A class for sorting an unordered container of pointers according to the sort
// order of an ordered container of pointers. Sorting is stable.
//
// Terminology:
// - unmapped element: An element from the unordered container that does not
//   have a corresponding element in the ordered container.
template <typename PointedToTy>
class MappedPtrContainerSorter {
 public:
  // A function to map elements from an ordered container to elements in an
  // unordered container. Not every element in ordered_container need map to an
  // element in unordered_container and vice versa.
  using MapPtrFn = absl::FunctionRef<const PointedToTy*(const PointedToTy*)>;

  // A function that maps unmapped elements (from an unordered container) to an
  // index in the final sorted result. The returned index indicates that the
  // unmapped element should be placed just after the mapped element at that
  // index, in the result without unmapped elements. See
  // IndexBeforeMappedElementsFn() and IndexAfterMappedElementsFn() for how to
  // indicate that an unmapped element should be placed before or after all
  // mapped elements, respectively. Unmapped elements destined for the same
  // index will retain their order from the unordered container.
  using UnmappedPtrIndexFn = absl::FunctionRef<size_t(const PointedToTy*)>;

  // Functions that return an UnmappedElementIndexFn that indicates that
  // ummapped elements (from an unordered container) should be placed before or
  // after all mapped elements, respectively.
  static UnmappedPtrIndexFn IndexBeforeMappedElementsFn();
  static UnmappedPtrIndexFn IndexAfterMappedElementsFn();

  // Returned function always returns an error.
  static UnmappedPtrIndexFn InvalidIndexFn();

  // Sorts an unordered container of pointers according to the order of an
  // ordered container of pointers. Sorting is stable. Works with POD pointers,
  // const POD pointers, and unique_ptrs. If an error is returned,
  // unordered_container is not modified. Returns an error status if:
  // - unmapped_index() returns an invalid index
  // - An internal error occurs. (This should theoretically not happen.)
  template <typename OrderedTy, typename UnorderedTy>
  static Status Sort(MapPtrFn map_ptr, UnmappedPtrIndexFn unmapped_index,
                     const OrderedTy& ordered_container,
                     UnorderedTy& unordered_container);

 private:
  // A class for sorting the indices of the unordered_container.
  class SortedIndices {
   public:
    // max_partial_order_exclusive is 1 greater than the maximum partial order
    // value allowed to be sent to AddMappedElement().
    SortedIndices(size_t max_partial_order_exclusive,
                  size_t unordered_container_size)
        : max_partial_order_exclusive_(max_partial_order_exclusive),
          unordered_container_size_(unordered_container_size),
          mapped_element_indices_by_partial_order_(
              max_partial_order_exclusive) {}

    // Specify the partial ordering value of a mapped element from the
    // unordered container. The partial ordering is amongst other mapped
    // elements.
    Status AddMappedElement(size_t unordered_container_index,
                            size_t partial_order);

    // Specify the index (amongst mapped elements), where an unmapped element
    // should be inserted. The unmapped element is inserted just after the
    // mapped element with index target_index_amongst_mapped_elements.
    void AddUnmappedElement(size_t unordered_container_index,
                            size_t target_index_amongst_mapped_elements);

    std::string ToString() const;

    // The result maps each element in the unordered_container to the target
    // index that it will occupy in the sorted result.
    StatusOr<std::vector<size_t>> Flatten() const;

   private:
    SortedIndices() = delete;

    size_t max_partial_order_exclusive_;
    size_t unordered_container_size_;
    std::vector<std::vector<size_t>> mapped_element_indices_by_partial_order_;
    absl::flat_hash_map<size_t, std::vector<size_t>>
        target_index_to_unmapped_element_index_;
  };

  static size_t IndexBeforeMappedElements() {
    return std::numeric_limits<size_t>::max() - 2;
  }

  static size_t IndexAfterMappedElements() {
    return std::numeric_limits<size_t>::max() - 1;
  }

  static size_t InvalidIndex() { return std::numeric_limits<size_t>::max(); }

  // Returns a mapping in which the element at index i indicates the target
  // index that unordered_container[i] should occupy in the sorted result.
  template <typename OrderedTy, typename UnorderedTy>
  static StatusOr<std::vector<size_t>> ComputeNewIndices(
      MapPtrFn map_ptr, UnmappedPtrIndexFn unmapped_index,
      const OrderedTy& ordered_container,
      const UnorderedTy& unordered_container);

  // Reorders unordered_container according to the indices in new_indices. See
  // ComputeNewIndices() for how to interpret new_indices.
  template <typename UnorderedTy>
  static void Reorder(std::vector<size_t> new_indices,
                      UnorderedTy& unordered_container);
};

///// Template implementation below /////

namespace mapped_ptr_container_sorter_internal {

template <typename I, typename O>
struct PtrGetter {
  // Extracts a pointer of type O from i.
  static O Get(I i);
};

template <typename T>
struct PtrGetter<T* const&, const T*> {
  static const T* Get(T* const& p) { return p; }
};

template <typename T>
struct PtrGetter<T const* const&, const T*> {
  static const T* Get(T const* const& p) { return p; }
};

template <typename T>
struct PtrGetter<T*&, T*> {
  static T* Get(T*& p) { return p; }
};

template <typename T>
struct PtrGetter<const std::unique_ptr<T>&, const T*> {
  static const T* Get(const std::unique_ptr<T>& p) { return p.get(); }
};

template <typename T>
struct PtrGetter<std::unique_ptr<T>&, T*> {
  static T* Get(std::unique_ptr<T>& p) { return p.get(); }
};

}  // namespace mapped_ptr_container_sorter_internal

template <typename PointedToTy>
typename MappedPtrContainerSorter<PointedToTy>::UnmappedPtrIndexFn
MappedPtrContainerSorter<PointedToTy>::IndexBeforeMappedElementsFn() {
  static const auto fn = [](const PointedToTy*) {
    return IndexBeforeMappedElements();
  };
  return fn;
}

template <typename PointedToTy>
typename MappedPtrContainerSorter<PointedToTy>::UnmappedPtrIndexFn
MappedPtrContainerSorter<PointedToTy>::IndexAfterMappedElementsFn() {
  static const auto fn = [](const PointedToTy*) {
    return IndexAfterMappedElements();
  };
  return fn;
}

template <typename PointedToTy>
typename MappedPtrContainerSorter<PointedToTy>::UnmappedPtrIndexFn
MappedPtrContainerSorter<PointedToTy>::InvalidIndexFn() {
  static const auto fn = [](const PointedToTy*) { return InvalidIndex(); };
  return fn;
}

template <typename PointedToTy>
Status MappedPtrContainerSorter<PointedToTy>::SortedIndices::AddMappedElement(
    size_t unordered_container_index, size_t partial_order) {
  if (partial_order >= mapped_element_indices_by_partial_order_.size()) {
    return InternalStrCat("invalid partial order: ", partial_order, " v max(",
                          mapped_element_indices_by_partial_order_.size(), ")");
  }

  mapped_element_indices_by_partial_order_[partial_order].push_back(
      unordered_container_index);
  return OkStatus();
}

template <typename PointedToTy>
void MappedPtrContainerSorter<PointedToTy>::SortedIndices::AddUnmappedElement(
    size_t unordered_container_index,
    size_t target_index_amongst_mapped_elements) {
  target_index_to_unmapped_element_index_[target_index_amongst_mapped_elements]
      .push_back(unordered_container_index);
}

template <typename PointedToTy>
std::string MappedPtrContainerSorter<PointedToTy>::SortedIndices::ToString()
    const {
  std::vector<std::string> mapped_element_strs;
  mapped_element_strs.reserve(mapped_element_indices_by_partial_order_.size());
  for (const auto& indices : mapped_element_indices_by_partial_order_) {
    mapped_element_strs.push_back(
        absl::StrCat("[", absl::StrJoin(indices, ", "), "]"));
  }
  std::vector<std::string> unmapped_element_strs;
  unmapped_element_strs.reserve(target_index_to_unmapped_element_index_.size());
  for (const auto& kv : target_index_to_unmapped_element_index_) {
    std::string key = absl::StrCat(kv.first);
    if (kv.first == IndexBeforeMappedElements()) {
      key = "before_mapped";
    }
    if (kv.first == IndexAfterMappedElements()) {
      key = "after_mapped";
    }
    if (kv.first == InvalidIndex()) {
      key = "invalid";
    }
    unmapped_element_strs.push_back(
        absl::StrCat(key, ": [", absl::StrJoin(kv.second, ", "), "]"));
  }

  return absl::StrCat(
      "max_partial_order_exclusive_: ", max_partial_order_exclusive_, "\n",
      "unordered_container_size_: ", unordered_container_size_, "\n",
      "mapped_element_indices_by_partial_order_: [",
      absl::StrJoin(mapped_element_strs, ", "), "]\n",
      "target_index_to_unmapped_element_index_: {",
      absl::StrJoin(unmapped_element_strs, ", "), "}\n");
}

template <typename PointedToTy>
StatusOr<std::vector<size_t>>
MappedPtrContainerSorter<PointedToTy>::SortedIndices::Flatten() const {
  std::vector<size_t> result(unordered_container_size_, InvalidIndex());
  size_t next_available_index = 0;
  auto next_index_fn = [&]() -> StatusOr<size_t> {
    if (next_available_index >= unordered_container_size_) {
      return InternalStrCat(
          "invalid unordered_container index: ", next_available_index,
          " v size(", unordered_container_size_, ")");
    }
    return next_available_index++;
  };

  if (target_index_to_unmapped_element_index_.contains(
          IndexBeforeMappedElements())) {
    const auto& indices =
        target_index_to_unmapped_element_index_.at(IndexBeforeMappedElements());
    for (size_t index : indices) {
      TF_ASSIGN_OR_RETURN(result[index], next_index_fn());
    }
  }
  size_t num_inserted_mapped_elements = 0;
  for (const auto& mapped_element_indices :
       mapped_element_indices_by_partial_order_) {
    for (size_t mapped_element_index : mapped_element_indices) {
      TF_ASSIGN_OR_RETURN(result[mapped_element_index], next_index_fn());
      ++num_inserted_mapped_elements;
      if (target_index_to_unmapped_element_index_.contains(
              num_inserted_mapped_elements - 1)) {
        const auto& unmapped_element_indices =
            target_index_to_unmapped_element_index_.at(
                num_inserted_mapped_elements - 1);
        for (size_t unmapped_element_index : unmapped_element_indices) {
          TF_ASSIGN_OR_RETURN(result[unmapped_element_index], next_index_fn());
        }
      }
    }
  }
  if (target_index_to_unmapped_element_index_.contains(
          IndexAfterMappedElements())) {
    const auto& indices =
        target_index_to_unmapped_element_index_.at(IndexAfterMappedElements());
    for (size_t index : indices) {
      TF_ASSIGN_OR_RETURN(result[index], next_index_fn());
    }
  }

  // Ensure that every element in unordered_container has a valid new index.
  absl::flat_hash_set<size_t> used_indices;
  for (size_t index : result) {
    if (used_indices.contains(index)) {
      return InternalStrCat(
          "2 elements in unordered_container are destined for the same "
          "index: ",
          index);
    }
    if (index >= unordered_container_size_) {
      return InvalidArgumentStrCat("invalid unordered_container index: ", index,
                                   " v size(", unordered_container_size_, ")");
    }
  }

  return result;
}

template <typename PointedToTy>
template <typename OrderedTy, typename UnorderedTy>
StatusOr<std::vector<size_t>>
MappedPtrContainerSorter<PointedToTy>::ComputeNewIndices(
    MapPtrFn map_ptr, UnmappedPtrIndexFn unmapped_index,
    const OrderedTy& ordered_container,
    const UnorderedTy& unordered_container) {
  using UnorderedPtrGetter = mapped_ptr_container_sorter_internal::PtrGetter<
      typename UnorderedTy::const_reference, const PointedToTy*>;
  using OrderedPtrGetter = mapped_ptr_container_sorter_internal::PtrGetter<
      typename OrderedTy::const_reference, const PointedToTy*>;

  if (unordered_container.size() >= IndexBeforeMappedElements()) {
    return InvalidArgumentStrCat("Unordered container is too large to sort.");
  }

  // Step 1: build a set of the ptrs in unordered_container
  absl::flat_hash_set<const PointedToTy*> unordered_ptrs;
  for (const auto& unordered_element : unordered_container) {
    const PointedToTy* ptr = UnorderedPtrGetter::Get(unordered_element);
    unordered_ptrs.insert(ptr);
  }

  // Step 2: for mapped elements (in unordered_container), create a map from
  // mapped ptr -> partial ordering
  absl::flat_hash_map<const PointedToTy*, std::list<size_t>>
      mapped_ptr_to_partial_order;
  size_t next_partial_order_value = 0;
  for (const auto& ordered_element : ordered_container) {
    const PointedToTy* ordered_ptr = OrderedPtrGetter::Get(ordered_element);
    const PointedToTy* unordered_ptr = map_ptr(ordered_ptr);
    if (!unordered_ptr) {
      // A corresponding unordered element does not exist.
      continue;
    }
    if (!unordered_ptrs.contains(unordered_ptr)) {
      // A pointer exists that maps to the ordered element, but it's not in our
      // unordered_container.
      continue;
    }
    mapped_ptr_to_partial_order[unordered_ptr].push_back(
        next_partial_order_value);
    ++next_partial_order_value;
  }

  // Step 3: create sorted unordered element indices
  SortedIndices result(next_partial_order_value, unordered_container.size());
  for (size_t i = 0; i < unordered_container.size(); ++i) {
    const PointedToTy* ptr = UnorderedPtrGetter::Get(unordered_container[i]);
    if (!mapped_ptr_to_partial_order.contains(ptr)) {
      // ptr is unmapped
      result.AddUnmappedElement(i, unmapped_index(ptr));
      continue;
    }

    // ptr is mapped
    //
    // Potentially, several elements in ordered_container map to ptr.
    // We assign ptr theindex corresponding to the next such ordered element.
    auto& index_list = mapped_ptr_to_partial_order[ptr];
    TF_RETURN_IF_ERROR(result.AddMappedElement(i, index_list.front()));
    // Do not map more than one unordered element to the same index, unless we
    // have no choice.
    if (index_list.size() > 1) {
      // We never remove the last ordered index, in case ptr appears in the
      // unordered_container more times than the ordered container.
      index_list.pop_front();
    }
  }

  VLOG(5) << "Pre flatten unordered_container result:\n" << result.ToString();
  return result.Flatten();
}

template <typename PointedToTy>
template <typename UnorderedTy>
void MappedPtrContainerSorter<PointedToTy>::Reorder(
    std::vector<size_t> new_indices, UnorderedTy& unordered_container) {
  size_t old_pos = 0;
  while (old_pos < new_indices.size()) {
    size_t new_pos = new_indices[old_pos];
    if (old_pos == new_pos) {
      ++old_pos;
      continue;
    }
    std::swap(new_indices[old_pos], new_indices[new_pos]);
    std::swap(unordered_container[old_pos], unordered_container[new_pos]);
  }
}

template <typename PointedToTy>
template <typename OrderedTy, typename UnorderedTy>
Status MappedPtrContainerSorter<PointedToTy>::Sort(
    MapPtrFn map_ptr, UnmappedPtrIndexFn unmapped_index,
    const OrderedTy& ordered_container, UnorderedTy& unordered_container) {
  std::vector<size_t> indices;
  TF_ASSIGN_OR_RETURN(
      indices, ComputeNewIndices(map_ptr, unmapped_index, ordered_container,
                                 unordered_container));
  Reorder(std::move(indices), unordered_container);
  return OkStatus();
}

}  // namespace xla

#endif  // XLA_SERVICE_MAPPED_PTR_CONTAINER_SORTER_H_
