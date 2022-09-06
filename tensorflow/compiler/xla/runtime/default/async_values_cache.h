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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_DEFAULT_ASYNC_VALUES_CACHE_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_DEFAULT_ASYNC_VALUES_CACHE_H_

#include "absl/synchronization/mutex.h"
#include "llvm/ADT/DenseMap.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime

namespace xla {
namespace runtime {

using tfrt::AsyncValue;
using tfrt::AsyncValuePtr;
using tfrt::AsyncValueRef;
using tfrt::Chain;
using tfrt::MakeConstructedAsyncValueRef;
using tfrt::MakeUnconstructedAsyncValueRef;

template <typename Key, typename Value>
class AsyncValuesCache {
 public:
  struct Entry;

  AsyncValuesCache() = default;

  // Returns a pointer to the cached value if it exists, otherwise returns
  // nullptr. It is the caller's responsibility to form an async reference and
  // extend its lifetime if the lifetime of the cached async value can be
  // larger than the lifetime of the cache.
  AsyncValuePtr<Value> Find(Key key) const;

  // Allocates an async value in the unconstructed state to store the cached
  // value with the given key.
  //
  // The `entry.allocated` value is `true` if the new async value was allocated,
  // and the caller is responsible for eventually setting the error or emplacing
  // the value. If it is false, then it means that the storage was already
  // allocated, and someone else will eventually update it.
  //
  // The returned `entry.size` value is equal to the size of the cache. If the
  // new async value was allocated, it will be reflected in the size.
  Entry Allocate(Key key);

  // Returns an async value that becomes available once all entries added to
  // the cache are available.
  AsyncValueRef<Chain> AllAvailable() const;

  struct Entry {
    AsyncValuePtr<Value> ptr;
    bool allocated;
    size_t size;
  };

 private:
  mutable absl::Mutex mu_;
  llvm::DenseMap<Key, AsyncValueRef<Value>> cache_ ABSL_GUARDED_BY(mu_);
};

template <typename Key, typename Value>
AsyncValuePtr<Value> AsyncValuesCache<Key, Value>::Find(Key key) const {
  absl::MutexLock lock(&mu_);
  auto it = cache_.find(key);
  return it != cache_.end() ? it->getSecond().AsPtr() : AsyncValuePtr<Value>();
}

template <typename Key, typename Value>
auto AsyncValuesCache<Key, Value>::Allocate(Key key) -> Entry {
  absl::MutexLock lock(&mu_);
  auto it = cache_.find(key);
  if (it != cache_.end())
    return {it->getSecond().AsPtr(), false, cache_.size()};

  AsyncValueRef<Value> allocated = MakeUnconstructedAsyncValueRef<Value>();

  auto emplaced = cache_.try_emplace(key, std::move(allocated));
  assert(emplaced.second && "emplace must be successful");
  return {emplaced.first->getSecond().AsPtr(), true, cache_.size()};
}

template <typename Key, typename Value>
AsyncValueRef<Chain> AsyncValuesCache<Key, Value>::AllAvailable() const {
  absl::MutexLock lock(&mu_);

  llvm::SmallVector<AsyncValue*> avs;
  for (auto& it : cache_) avs.push_back(it.getSecond().GetAsyncValue());

  AsyncValueRef<Chain> chain = MakeConstructedAsyncValueRef<Chain>();
  RunWhenReady(avs, [chain]() { chain.SetStateConcrete(); });
  return chain;
}

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_DEFAULT_ASYNC_VALUES_CACHE_H_
