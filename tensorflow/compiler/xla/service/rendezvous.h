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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_RENDEZVOUS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_RENDEZVOUS_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace xla {

template <typename K, typename V>
class ThreadSafeMap {
 public:
  V& operator[](const K& key) {
    absl::MutexLock lock(&mutex_);
    std::unique_ptr<V>& value = map_[key];
    if (value == nullptr) value = std::make_unique<V>();
    return *value;
  }

  void ForEachValue(absl::FunctionRef<void(V&)> fn) {
    absl::MutexLock lock(&mutex_);
    for (const auto& [_, value] : map_) fn(*value);
  }

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<K, std::unique_ptr<V>> map_ ABSL_GUARDED_BY(mutex_);
};

void AwaitAndLogIfStuck(absl::Mutex& mutex, const absl::Condition& condition,
                        absl::Duration warn_stuck_timeout,
                        absl::Duration terminate_timeout);

// A rendezvous for a group of threads.
//
// The group of threads identifies itself with a key that must be unique to the
// the group. When all threads have arrived at the rendezvous, one thread
// executes the given function with the values supplied by each thread, and all
// threads receive the result.
// TODO(cjfj): Replace XLA rendezvous code with this simpler implementation.
template <typename R, typename K, typename V>
std::shared_ptr<R> RendezvousSingle(
    const K& key, const V& value, size_t num_threads,
    absl::FunctionRef<R(absl::Span<const V* const>)> fn,
    absl::Duration warn_stuck_timeout = absl::InfiniteDuration(),
    absl::Duration terminate_timeout = absl::InfiniteDuration()) {
  // Fast-path (DO NOT REMOVE: the logic below doesn't work for single thread).
  if (num_threads == 1) return std::make_shared<R>(fn({&value}));

  struct State {
    absl::Mutex mutex;
    std::vector<const V*> values ABSL_GUARDED_BY(mutex);
    std::shared_ptr<R> result ABSL_GUARDED_BY(mutex);
  };

  static auto& states = *new ThreadSafeMap<K, State>;
  State& state = states[key];

  absl::MutexLock lock(&state.mutex);
  state.values.push_back(&value);

  std::shared_ptr<R> result;
  if (state.values.size() == num_threads) {
    // Last thread to arrive executes the function.
    CHECK(state.result == nullptr);
    result = std::make_shared<R>(fn(state.values));
    state.result = result;
    state.values.clear();
  } else {
    absl::Condition result_ready(
        +[](std::shared_ptr<R>* ptr) { return ptr->get() != nullptr; },
        &state.result);
    AwaitAndLogIfStuck(state.mutex, result_ready, warn_stuck_timeout,
                       terminate_timeout);

    // There is one use of the result in the shared state, plus one use for each
    // thread that has already retrieved the result.
    if (state.result.use_count() < num_threads) {
      result = state.result;
    } else {
      // Last thread to retrieve the result takes the result from the state,
      // allowing the other threads to exit the function.
      return std::move(state.result);
    }
  }

  // Wait for all threads to have retrieved the result. Without this, a thread
  // could duplicate or delete its copy of the result, invalidating the use
  // count logic above.
  absl::Condition result_taken(
      +[](std::shared_ptr<R>* ptr) { return ptr->get() == nullptr; },
      &state.result);
  AwaitAndLogIfStuck(state.mutex, result_taken, warn_stuck_timeout,
                     terminate_timeout);
  return result;
}

// A rendezvous for a group of threads.
//
// The group of threads identifies itself with a key that must be unique to the
// the group. When all threads have arrived at the rendezvous, one thread
// executes the given function and all threads receive the result.
// TODO(cjfj): Replace XLA rendezvous code with this simpler implementation.
template <typename R, typename K>
std::shared_ptr<R> RendezvousSingle(
    const K& key, size_t num_threads, absl::FunctionRef<R()> fn,
    absl::Duration warn_stuck_timeout = absl::InfiniteDuration(),
    absl::Duration terminate_timeout = absl::InfiniteDuration()) {
  // Pass an arbitrary value that is ignored.
  return RendezvousSingle<R, K, int>(
      key, 0, num_threads, [fn](absl::Span<const int* const>) { return fn(); },
      warn_stuck_timeout, terminate_timeout);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_RENDEZVOUS_H_
