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

#ifndef XLA_SERVICE_RENDEZVOUS_H_
#define XLA_SERVICE_RENDEZVOUS_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tsl/platform/logging.h"

namespace xla {

//===----------------------------------------------------------------------===//
// A rendezvous for a group of threads.
//===----------------------------------------------------------------------===//

// A little bit of compile time metaprogramming to simplify the rendezvous
// return type for functions returning `absl::StatusOr`. If we detect that
// rendezvous callback returns `absl::StatusOr` we swap the order of a shared
// pointer and status container.

template <typename R>
struct RendezvousResult {
  using Type = std::shared_ptr<R>;

  static Type Wrap(R result) { return std::make_shared<R>(std::move(result)); }
};

template <typename R>
struct RendezvousResult<absl::StatusOr<R>> {
  using Type = absl::StatusOr<std::shared_ptr<R>>;

  static Type Wrap(absl::StatusOr<R> result) {
    if (!result.ok()) return result.status();
    return std::make_shared<R>(std::move(*result));
  }
};

template <typename R>
using RendezvousResultType = typename RendezvousResult<R>::Type;

// The group of threads identifies itself with a key that must be unique to
// the the group. When all threads have arrived at the rendezvous, one thread
// executes the given function with the values supplied by each thread, and
// all threads receive the result. Rendezvous must have a human readable name to
// make easy to debug stuck and timed out attempts.
template <typename R, typename K, typename V, typename Fn>
RendezvousResultType<R> RendezvousSingle(
    std::string_view name, const K& key, const V& value, size_t num_threads,
    Fn fn, absl::Duration warn_stuck_timeout = absl::InfiniteDuration(),
    absl::Duration terminate_timeout = absl::InfiniteDuration());

// A rendezvous for a group of threads that do not have any value arguments.
template <typename R, typename K, typename Fn>
RendezvousResultType<R> RendezvousSingle(
    std::string_view name, const K& key, size_t num_threads, Fn fn,
    absl::Duration warn_stuck_timeout = absl::InfiniteDuration(),
    absl::Duration terminate_timeout = absl::InfiniteDuration());

// A rendezvous for a group of threads that do not have any computation to run
// and simply acts as a barrier for a group of thread.
template <typename K>
void RendezvousSingle(
    std::string_view name, const K& key, size_t num_threads,
    absl::Duration warn_stuck_timeout = absl::InfiniteDuration(),
    absl::Duration terminate_timeout = absl::InfiniteDuration());

//===----------------------------------------------------------------------===//
// Internal implementation details.
//===----------------------------------------------------------------------===//

namespace internal {

// A state for a single round of rendezvous. We expect exactly `num_treads` to
// arrive to a rendezvous and update corresponding slots in `values`. We
// pre-allocate storage for values so at run time each participant doesn't have
// to grab a lock and can simple write to the destination storage.
template <typename R, typename V>
struct RendezvousState {
  explicit RendezvousState(size_t num_threads)
      : ack(0), rel(0), values(num_threads, nullptr), result(nullptr) {}

  std::atomic<int32_t> ack;
  std::atomic<int32_t> rel;
  std::vector<const V*> values;

  absl::Notification ready;  // signals availability of `result`
  RendezvousResultType<R> result;
};

// A container for in-progress rendezvous.
//
// Rendezvous state ownership:
//
// (1) When rendezvous participant initiates a rendezvous with a particular key
//     we create a new state for it, keep it in a map for tracking and return a
//     shared pointer to the caller.
//
// (2) When rendezvous participant joins in-progress rendezvous it gets back
//     a shared pointer that is copied from a tracking map.
//
// (3) When the last rendezvous participant computes the result it completes the
//     rendezvous and removes a shared pointer to a state. Remaining shared
//     pointers destructed when all participants are notified.
//
// This process guarantees that all completed rendezvous are removed from a map
// and a map has records only for rendezvous in progress.
template <typename K, typename R, typename V>
class RendezvousMap {
 public:
  using State = RendezvousState<R, V>;

  std::shared_ptr<State> Join(const K& key, size_t num_threads) {
    absl::MutexLock lock(&mutex_);
    std::shared_ptr<State>& state = state_[key];

    // Join an in-progress rendezvous.
    if (state) return state;

    // Join a newly created rendezvous.
    return state = std::make_shared<State>(num_threads);
  }

  void Complete(const K& key, RendezvousResultType<R> result) {
    std::shared_ptr<State> state = [&] {
      absl::MutexLock lock(&mutex_);

      // Extract state from the map so we can immediately start a new round of
      // rendezvous with the same key. A state for previous rendezvous will be
      // destructed with the last copy of a shared pointer.
      std::shared_ptr<State> state = state_.extract(key).mapped();

      // Check that we have have exactly the number of participants we expected:
      // +1 reference for all participants and a +1 reference we extracted.
      CHECK_EQ(state.use_count(), 1 + state->values.size());  // NOLINT

      return state;
    }();

    // Notify awaiting participants without holding a lock.
    state->result = std::move(result);
    state->ready.Notify();
  }

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<K, std::shared_ptr<State>> state_ ABSL_GUARDED_BY(mutex_);
};

void AwaitAndLogIfStuck(absl::Notification& ready, std::string_view name,
                        size_t num_threads, absl::Duration warn_stuck_timeout,
                        absl::Duration terminate_timeout);
}  // namespace internal

//===----------------------------------------------------------------------===//
// Rendezvous implemenetation.
//===----------------------------------------------------------------------===//

template <typename R, typename K, typename V, typename Fn>
RendezvousResultType<R> RendezvousSingle(std::string_view name, const K& key,
                                         const V& value, size_t num_threads,
                                         Fn fn,
                                         absl::Duration warn_stuck_timeout,
                                         absl::Duration terminate_timeout) {
  // Check that `fn` is callable with a span of values and returns `R`.
  static_assert(std::is_invocable_r_v<R, Fn, absl::Span<const V*>>,
                "invalid rendezvous function signature");

  // Fast-path (DO NOT REMOVE: the logic below doesn't work for single thread).
  if (num_threads == 1) {
    const V* ptr = &value;
    return RendezvousResult<R>::Wrap(fn(absl::MakeSpan(&ptr, 1)));
  }

  using State = internal::RendezvousState<R, V>;
  static auto& rendezvous = *new internal::RendezvousMap<K, R, V>;
  std::shared_ptr<State> state = rendezvous.Join(key, num_threads);

  // If we got an id larger than `num_threads` it means that we have multiple
  // rendezvous sharing the same key running concurrently.
  int64_t id = state->ack.fetch_add(1);
  CHECK_LT(id, num_threads)  // NOLINT
      << "Id can't be larger than the number of participating threads"
      << "; id=" << id << "; num_threads=" << num_threads;

  state->values[id] = &value;

  // Use a second atomic to safely publish values without data races.
  if constexpr (!std::is_same_v<R, std::nullopt_t>) {
    id = state->rel.fetch_add(1);
  }

  if (id < num_threads - 1) {
    // Threads arriving before the last one wait for a result to be computed by
    // the last joining thread.
    internal::AwaitAndLogIfStuck(state->ready, name, num_threads,
                                 warn_stuck_timeout, terminate_timeout);
  } else {
    // Last thread to arrive executes the function and completes rendezvous by
    // making result available to all participants. All other participants will
    // be notified via `state->ready` notification when result is ready, and we
    // rely on the notification to create a memory barrier that makes access to
    // `state->result` safe without any extra synchronization.
    rendezvous.Complete(key, RendezvousResult<R>::Wrap(fn(state->values)));
  }

  return state->result;
}

template <typename R, typename K, typename Fn>
RendezvousResultType<R> RendezvousSingle(std::string_view name, const K& key,
                                         size_t num_threads, Fn fn,
                                         absl::Duration warn_stuck_timeout,
                                         absl::Duration terminate_timeout) {
  return RendezvousSingle<R, K, std::nullopt_t>(
      name, key, std::nullopt, num_threads, [fn](auto) { return fn(); },
      warn_stuck_timeout, terminate_timeout);
}

template <typename K>
void RendezvousSingle(std::string_view name, const K& key, size_t num_threads,
                      absl::Duration warn_stuck_timeout,
                      absl::Duration terminate_timeout) {
  RendezvousSingle<std::nullopt_t, K, std::nullopt_t>(
      name, key, std::nullopt, num_threads, [](auto) { return std::nullopt; },
      warn_stuck_timeout, terminate_timeout);
}

}  // namespace xla

#endif  // XLA_SERVICE_RENDEZVOUS_H_
