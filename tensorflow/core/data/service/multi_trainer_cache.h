/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_MULTI_TRAINER_CACHE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_MULTI_TRAINER_CACHE_H_

#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/logging_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// Sliding-window cache shared across concurrent trainers. Readers call `Get` to
// read elements they haven't read. After a trainer reads an element, it remains
// in the cache and the data is shared with other trainers. This is useful for
// datasets involving expensive computation, and multiple models use the same
// data for training. For example, for hyperparameter tuning.
//
// The cache progresses when a trainer that has consumed all elements in the
// cache requests additional data. It has a bounded size. Elements are garbage
// collected when the cache becomes full. Consequently, trainers read from a
// sliding window through the dataset and may not read the full dataset.
//
// This class is thread-safe.
//
// Example usage:
//
//   // Assumes `get_next_fn` returns 1, 2, 3, ...
//   MultiTrainerCache<int64_t> cache(
//       /*max_cache_size_bytes=*/10 * (size_t{1} << 30),  // 10GB
//       get_next_fn, get_element_size_bytes_fn);
//
//   std::shared_ptr<const int64_t> next;
//   TF_ASSIGN_OR_RETURN(next, cache.Get("Trainer 1"));  // Returns 1
//   TF_ASSIGN_OR_RETURN(next, cache.Get("Trainer 2"));  // Returns 1
//   TF_ASSIGN_OR_RETURN(next, cache.Get("Trainer 1"));  // Returns 2
//   TF_ASSIGN_OR_RETURN(next, cache.Get("Trainer 2"));  // Returns 2
template <class ElementType>
class MultiTrainerCache {
 public:
  using GetElementFn = std::function<StatusOr<ElementType>()>;
  using GetElementSizeBytesFn = std::function<size_t(const ElementType&)>;

  // Creates a `MultiTrainerCache` with `max_cache_size_bytes` of memory budget.
  // The cache should be able to hold at least one element, i.e.:
  // REQUIRES: max_cache_size_bytes >= max(get_element_size(*))
  // TODO(b/221104308): Use an interface to encapsulate the function inputs.
  explicit MultiTrainerCache(size_t max_cache_size_bytes, GetElementFn get_next,
                             GetElementSizeBytesFn get_element_size);
  virtual ~MultiTrainerCache() = default;

  // Gets the next element for `trainer`. A `trainer_id` identifies the trainer
  // reading from the cache. If one trainer has read data, the data is shared
  // with other trainers.
  StatusOr<std::shared_ptr<const ElementType>> Get(
      const std::string& trainer_id);

  // Cancels the cache with `status` and notifies the readers. After cancelling,
  // all `Get` calls will return `status`.
  // REQUIRES: !status.ok()
  void Cancel(Status status);

  // Returns true if the cache has been cancelled.
  bool IsCancelled() const;

 private:
  // Returns true if element is ready for `trainer_id`. An element is ready if
  // other trainers have read the data and the data remains in the cache. If the
  // data is not ready, one of the trainers need to extend the cache.
  bool IsElementReady(const std::string& trainer_id);

  // Returns the absolute element index relative to the dataset (not relative to
  // the cached elements).
  size_t GetElementIndex(const std::string& trainer_id);

  // Returns the next element for `trainer_id`.
  StatusOr<std::shared_ptr<const ElementType>> GetElement(
      const std::string& trainer_id);

  // Reads a new element and writes it into the cache.
  Status ExtendCache();

  // Frees old elements to keep the cache size below `max_cache_size_bytes_`.
  // `new_element_size_bytes` is the size of the new element being inserted.
  void FreeSpace(size_t new_element_size_bytes);

  // Maximum cache size in bytes.
  const size_t max_cache_size_bytes_;
  const GetElementFn get_next_;
  const GetElementSizeBytesFn get_element_size_bytes_;

  mutable mutex mu_;
  mutable condition_variable cv_;

  // If `status_` is non-OK, the cache is cancelled, and all method calls will
  // return this status.
  Status status_ TF_GUARDED_BY(mu_) = Status::OK();

  // `cache_` stores the cached elements.
  std::deque<std::shared_ptr<const ElementType>> cache_ TF_GUARDED_BY(mu_);
  size_t cache_size_bytes_ TF_GUARDED_BY(mu_) = 0;
  size_t cache_start_index_ TF_GUARDED_BY(mu_) = 0;

  // True if one thread is extending the cache.
  bool extending_cache_ TF_GUARDED_BY(mu_) = false;

  // Maps trainer IDs to element indices. The indices are absolute indices
  // within the dataset. The actual index to use with `cache_` would be
  // `trainer_to_element_index_map_[trainer_id] - cache_start_index_`.
  absl::flat_hash_map<std::string, size_t> trainer_to_element_index_map_
      TF_GUARDED_BY(mu_);
};

template <class ElementType>
MultiTrainerCache<ElementType>::MultiTrainerCache(
    size_t max_cache_size_bytes, GetElementFn get_next,
    GetElementSizeBytesFn get_element_size_bytes)
    : max_cache_size_bytes_(max_cache_size_bytes),
      get_next_(std::move(get_next)),
      get_element_size_bytes_(std::move(get_element_size_bytes)) {
  DCHECK_GT(max_cache_size_bytes, 0)
      << "MultiTrainerCache size must be greater than 0.";
  VLOG(2) << "Initialized tf.data service multi-trainer cache with "
          << FormatBytes(max_cache_size_bytes) << " of memory.";
}

template <class ElementType>
StatusOr<std::shared_ptr<const ElementType>>
MultiTrainerCache<ElementType>::Get(const std::string& trainer_id)
    TF_LOCKS_EXCLUDED(mu_) {
  if (trainer_id.empty()) {
    return errors::Internal(
        "tf.data service multi-trainer cache trainer ID must be non-empty.");
  }

  while (true) {
    bool should_extend_cache = false;
    {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(status_);
      if (IsElementReady(trainer_id)) {
        return GetElement(trainer_id);
      }

      // Extends the cache or waits for another thread to extend the cache. When
      // concurrent trainers wait for the next element, only one of them should
      // extend the cache.
      if (extending_cache_) {
        should_extend_cache = false;
        cv_.wait(l);
      } else {
        should_extend_cache = true;
        extending_cache_ = true;
      }
    }

    if (should_extend_cache) {
      TF_RETURN_IF_ERROR(ExtendCache());
    }
  }
}

template <class ElementType>
bool MultiTrainerCache<ElementType>::IsElementReady(
    const std::string& trainer_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return GetElementIndex(trainer_id) < cache_start_index_ + cache_.size();
}

template <class ElementType>
StatusOr<std::shared_ptr<const ElementType>>
MultiTrainerCache<ElementType>::GetElement(const std::string& trainer_id)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  size_t element_index = GetElementIndex(trainer_id);
  if (element_index >= std::numeric_limits<size_t>::max()) {
    return errors::Internal(
        "tf.data service caching element index exceeds integer limit. Got ",
        element_index);
  }

  std::shared_ptr<const ElementType> result =
      cache_[element_index - cache_start_index_];
  trainer_to_element_index_map_[trainer_id] = element_index + 1;
  return result;
}

template <class ElementType>
size_t MultiTrainerCache<ElementType>::GetElementIndex(
    const std::string& trainer_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  size_t element_index = trainer_to_element_index_map_[trainer_id];
  if (element_index < cache_start_index_) {
    element_index = cache_start_index_;
  }
  return element_index;
}

template <class ElementType>
Status MultiTrainerCache<ElementType>::ExtendCache() TF_LOCKS_EXCLUDED(mu_) {
  StatusOr<ElementType> element = get_next_();
  if (!element.ok()) {
    mutex_lock l(mu_);
    extending_cache_ = false;
    cv_.notify_all();
    return element.status();
  }

  const size_t new_element_size_bytes = get_element_size_bytes_(*element);
  if (new_element_size_bytes > max_cache_size_bytes_) {
    return errors::InvalidArgument(
        "tf.data service element size is larger than cache size in bytes. Got ",
        "element size: ", new_element_size_bytes,
        " and cache size: ", max_cache_size_bytes_);
  }

  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(status_);
  FreeSpace(new_element_size_bytes);
  cache_.push_back(std::make_shared<ElementType>(std::move(*element)));
  cache_size_bytes_ += new_element_size_bytes;
  extending_cache_ = false;
  cv_.notify_all();
  return Status::OK();
}

template <class ElementType>
void MultiTrainerCache<ElementType>::FreeSpace(size_t new_element_size_bytes)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  size_t num_elements_discarded = 0;
  while (!cache_.empty() &&
         cache_size_bytes_ + new_element_size_bytes > max_cache_size_bytes_) {
    size_t free_bytes = get_element_size_bytes_(*cache_.front());
    cache_.pop_front();
    cache_size_bytes_ -= free_bytes;
    ++cache_start_index_;
    ++num_elements_discarded;
  }

  VLOG(3) << "Freed " << num_elements_discarded << " element(s) from "
          << "tf.data service multi-trainer cache. Memory usage: "
          << FormatBytes(cache_size_bytes_) << ".";
}

template <class ElementType>
void MultiTrainerCache<ElementType>::Cancel(Status status)
    TF_LOCKS_EXCLUDED(mu_) {
  DCHECK(!status.ok())
      << "Cancelling MultiTrainerCache requires a non-OK status. Got "
      << status;
  VLOG(2) << "Cancel tf.data service multi-trainer cache with status "
          << status;
  mutex_lock l(mu_);
  status_ = std::move(status);
  cv_.notify_all();
}

template <class ElementType>
bool MultiTrainerCache<ElementType>::IsCancelled() const
    TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return !status_.ok();
}
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_MULTI_CLIENT_CACHE_H_
