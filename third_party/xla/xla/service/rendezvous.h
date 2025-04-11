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
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

//===----------------------------------------------------------------------===//
// Rendezvous synchronization.
//===----------------------------------------------------------------------===//

// Rendezvous is an XLA synchronization primitive that guarantees that all
// participating threads arrive to a rendezvous barrier identified by a key, and
// the last arriving thread becomes a leader that executes a rendezvous
// callback. The result of executing a callback broadcasted back to all
// participants as an `std::shared_ptr<R>` value, which makes all participants
// "collective owners" of the computed value.
//
// XLA uses rendezvous to guarantee that all ranks make progress together when
// executing a partitioned XLA program, and acts as a guard against the
// deadlocks in the lower parts of the stack (i.e. if not all participants
// arrive to NCCL collective, then we will get a deadlock on device, which is a
// lot harder to debug).
//
// Rendezvous can synchronize only within a same process, as it relies on
// shared memory to communicate between participants.
//
// If rendezvous reaches a `terminate_timeout`, it will return an error status
// to all participants, meaning that not all participants have arrived to the
// rendezvous barrier in the given time.
//
// Rendezvous callback must return the value of type `R`, or `absl::StatusOr<R>`
// which will be automatically converted to `absl::StatusOr<std::shared_ptr<R>>`
// for all participants.

//===----------------------------------------------------------------------===//
// Rendezvous API.
//===----------------------------------------------------------------------===//

// The group of threads identifies itself with a key that must be unique to
// the group. When all threads have arrived at the rendezvous, one thread
// executes the given function with the values supplied by each thread, and
// all threads receive the result. Rendezvous must have a human readable name to
// make easy to debug stuck and timed out attempts.
template <typename R, typename K, typename V, typename Fn>
absl::StatusOr<std::shared_ptr<R>> Rendezvous(
    absl::string_view name, const K& key, const V& value, size_t num_threads,
    Fn fn, absl::Duration warn_stuck_timeout = absl::InfiniteDuration(),
    absl::Duration terminate_timeout = absl::InfiniteDuration());

// A rendezvous for a group of threads that do not have any value arguments.
template <typename R, typename K, typename Fn>
absl::StatusOr<std::shared_ptr<R>> Rendezvous(
    absl::string_view name, const K& key, size_t num_threads, Fn fn,
    absl::Duration warn_stuck_timeout = absl::InfiniteDuration(),
    absl::Duration terminate_timeout = absl::InfiniteDuration());

// A rendezvous for a group of threads that do not have any computation to run
// and simply acts as a barrier for a group of thread.
template <typename K>
absl::Status Rendezvous(
    absl::string_view name, const K& key, size_t num_threads,
    absl::Duration warn_stuck_timeout = absl::InfiniteDuration(),
    absl::Duration terminate_timeout = absl::InfiniteDuration());

// An `std::once_flag`-like primitive for executing Rendezvous operations.
//
// RendezvousFlag guarantees that all or none participants in a rendezvous
// join the rendezvous process and once rendezvous is completed flag marked as
// `completed` and all further rendezvous using this flag will be skipped. It
// has a weaker than exactly-once guarantee and multiple racing rendezvous can
// execute in parallel, and the last completed rendezvous will switch flag to
// `completed` state.
//
// In XLA rendezvous are rare and used to guard costly shared state
// initialization, so in practice we do not expect to see many racing rendezvous
// and prefer simpler implementation with weaker guarantees.
//
// See: https://en.cppreference.com/w/cpp/thread/once_flag
class RendezvousFlag {
 public:
  RendezvousFlag();

  RendezvousFlag(const RendezvousFlag&) = delete;
  RendezvousFlag& operator=(const RendezvousFlag&) = delete;

  // RAII wrapper to exit from in-flight rendezvous when destructed.
  class InFlightRendezvous {
   public:
    explicit InFlightRendezvous(RendezvousFlag* flag);
    ~InFlightRendezvous();

    InFlightRendezvous(const InFlightRendezvous&) = delete;
    InFlightRendezvous& operator=(const InFlightRendezvous&) = delete;

    operator bool() const;  // NOLINT

   private:
    RendezvousFlag* flag_;
  };

  // Returns InFlightRendezvous convertible to `true` if the caller should join
  // the rendezvous process. If result conversion to bool is `false` it means
  // that the rendezvous is already completed.
  InFlightRendezvous TryJoin();

  bool IsCompleted() const;

 private:
  friend class InFlightRendezvous;

  std::atomic<int32_t> state_;
};

// A rendezvous for a group of threads that will be executed only if the flag is
// not in `completed` state and will switch it to `completed` after finishing a
// rendezvous. If rendezvous was not executed, the result will be an empty
// shared pointer.
template <typename R, typename K, typename Fn>
absl::StatusOr<std::shared_ptr<R>> Rendezvous(
    RendezvousFlag& flag, absl::string_view name, const K& key,
    size_t num_threads, Fn fn,
    absl::Duration warn_stuck_timeout = absl::InfiniteDuration(),
    absl::Duration terminate_timeout = absl::InfiniteDuration());

// A rendezvous for a group of threads that will be executed only if the flag is
// not in `completed` state and will switch it to `completed` after finishing a
// rendezvous.
template <typename K>
absl::Status Rendezvous(
    RendezvousFlag& flag, absl::string_view name, const K& key,
    size_t num_threads,
    absl::Duration warn_stuck_timeout = absl::InfiniteDuration(),
    absl::Duration terminate_timeout = absl::InfiniteDuration());

//===----------------------------------------------------------------------===//
// Internal implementation details.
//===----------------------------------------------------------------------===//

namespace internal {

// Detects types that are `absl::StatusOr<R>` container.
template <typename T>
struct IsStatusOrResult : std::false_type {};
template <typename T>
struct IsStatusOrResult<absl::StatusOr<T>> : std::true_type {};

// A base class for rendezvous state that holds synchronization primitives.
struct RendezvousStateSynchronization {
  explicit RendezvousStateSynchronization(size_t num_threads)
      : num_threads(num_threads), ack(0), rel(0), ready(false) {}

  int32_t num_threads;

  std::atomic<int32_t> ack;
  std::atomic<int32_t> rel;

  absl::Mutex mutex;
  absl::CondVar cv;

  // Signals availability of `result`.
  bool ready ABSL_GUARDED_BY(mutex);
};

// A state for a single round of rendezvous. We expect exactly `num_treads` to
// arrive to a rendezvous and update corresponding slots in `values`. We
// pre-allocate storage for values, so at run time each participant doesn't have
// to grab a lock and can simple write to the destination storage.
template <typename R, typename V>
struct RendezvousState : public RendezvousStateSynchronization {
  explicit RendezvousState(size_t n_threads)
      : RendezvousStateSynchronization(n_threads), values(n_threads, nullptr) {}

  std::vector<const V*> values;
  absl::StatusOr<std::shared_ptr<R>> result;
};

// A container for in-progress rendezvous.
//
// Rendezvous state ownership:
//
// (1) When rendezvous participant initiates a rendezvous with a particular key
//     we create a new state for it, keep it in a map as weak pointer for
//     tracking and return a shared pointer to the caller.
//
// (2) When rendezvous participant joins in-progress rendezvous it gets back
//     a shared pointer that is copied from a tracking map.
//
// (3) When rendezvous completes, the thread that completes it removes a state
//     from a map, so that the next rendezvous with the same key can start
//     immediately and create a new state.
//
// (4) If rendezvous failed to complete, the weak pointer will expire when all
//     participants left the rendezvous, and will be lazily garbage collected
//     in the next call to `Join`.
//
// This process guarantees that all completed rendezvous are removed from a map
// and a map has records only for rendezvous in progress.
template <typename K, typename R, typename V>
class RendezvousMap {
 public:
  using State = RendezvousState<R, V>;

  std::shared_ptr<State> Join(const K& key, size_t num_threads) {
    absl::MutexLock lock(&mutex_);

    // Erase expired rendezvous from the map.
    absl::erase_if(state_, [](const auto& e) { return e.second.expired(); });

    std::weak_ptr<State>& in_progress = state_[key];

    // Try to join an in-progress rendezvous for a given key.
    if (std::shared_ptr<State> joined = in_progress.lock()) {
      return joined;
    }

    // Start a new rendezvous for a given key.
    std::shared_ptr<State> start = std::make_shared<State>(num_threads);
    return (in_progress = start, start);
  }

  void Complete(const K& key) {
    absl::MutexLock lock(&mutex_);
    state_.erase(key);
  }

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<K, std::weak_ptr<State>> state_ ABSL_GUARDED_BY(mutex_);
};

void AwaitAndLogIfStuck(RendezvousStateSynchronization& state, int32_t id,
                        absl::string_view name,
                        absl::Duration warn_stuck_timeout,
                        absl::Duration terminate_timeout);

}  // namespace internal

//===----------------------------------------------------------------------===//
// Rendezvous implemenetation.
//===----------------------------------------------------------------------===//

template <typename R, typename V, typename Fn>
absl::StatusOr<std::shared_ptr<R>> InvokeRendezvous(
    Fn fn, absl::Span<const V*> values) {
  auto result = fn(values);

  if constexpr (internal::IsStatusOrResult<decltype(result)>::value) {
    if (ABSL_PREDICT_TRUE(result.ok())) {
      return std::make_shared<R>(*std::move(result));
    } else {
      return result.status();
    }
  } else {
    return std::make_shared<R>(std::move(result));
  }
}

template <typename R, typename K, typename V, typename Fn>
absl::StatusOr<std::shared_ptr<R>> Rendezvous(
    absl::string_view name, const K& key, const V& value, size_t num_threads,
    Fn fn, absl::Duration warn_stuck_timeout,
    absl::Duration terminate_timeout) {
  // Check that `fn` is callable with a span of values.
  static_assert(std::is_invocable_v<Fn, absl::Span<const V*>>,
                "invalid rendezvous function signature");

  // Fast-path (DO NOT REMOVE: the logic below doesn't work for single thread).
  if (num_threads == 1) {
    const V* ptr = &value;
    return InvokeRendezvous<R, V>(std::move(fn), absl::MakeSpan(&ptr, 1));
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

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        "Rendezvous",
        {{"num_threads", num_threads}, {"name", name}, {"id", id}});
  });

  // Signal all waiting threads that new participant has arrived.
  state->cv.SignalAll();

  // std::vector::operator[] creates data races, so we rely on data pointer
  // here and when we create an absl::Span below.
  *(state->values.data() + id) = &value;

  // Use a second atomic to safely publish values without data races.
  if constexpr (!std::is_same_v<R, std::nullopt_t>) {
    id = state->rel.fetch_add(1);
  }

  if (id < num_threads - 1) {
    // Threads arriving before the last one wait for a result to be computed by
    // the last joining thread.
    internal::AwaitAndLogIfStuck(*state, id, name, warn_stuck_timeout,
                                 terminate_timeout);
  } else {
    // Mark rendezvous as completed, so that we can immediately start a new
    // rendezvous with the same key.
    rendezvous.Complete(key);

    // Last thread to arrive executes the function and completes rendezvous by
    // making result available to all participants. All other participants will
    // be notified via `state->ready` flag when result is ready, and we rely on
    // the mutex to create a memory barrier that makes access to `state->result`
    // safe without any extra synchronization.
    tsl::profiler::TraceMe trace("InvokeRendezvous");
    absl::Span<const V*> values(state->values.data(), num_threads);

    // Check that we have have exactly the number of participants we expect.
    CHECK_EQ(state.use_count(), num_threads);  // NOLINT

    // Publish rendezvous result to all participants.
    state->result = InvokeRendezvous<R, V>(std::move(fn), values);

    // Switch `ready` flag to signal all participants that result is ready.
    {
      absl::MutexLock lock(&state->mutex);
      state->ready = true;
    }

    // Notify awaiting participants that result is ready.
    state->cv.SignalAll();
  }

  return state->result;
}

template <typename R, typename K, typename Fn>
absl::StatusOr<std::shared_ptr<R>> Rendezvous(
    absl::string_view name, const K& key, size_t num_threads, Fn fn,
    absl::Duration warn_stuck_timeout, absl::Duration terminate_timeout) {
  return Rendezvous<R, K, std::nullopt_t>(
      name, key, std::nullopt, num_threads, [fn](auto) { return fn(); },
      warn_stuck_timeout, terminate_timeout);
}

template <typename K>
absl::Status Rendezvous(absl::string_view name, const K& key,
                        size_t num_threads, absl::Duration warn_stuck_timeout,
                        absl::Duration terminate_timeout) {
  return Rendezvous<std::nullopt_t, K, std::nullopt_t>(
             name, key, std::nullopt, num_threads,
             [](auto) { return std::nullopt; }, warn_stuck_timeout,
             terminate_timeout)
      .status();
}

template <typename R, typename K, typename Fn>
absl::StatusOr<std::shared_ptr<R>> Rendezvous(
    RendezvousFlag& flag, absl::string_view name, const K& key,
    size_t num_threads, Fn fn, absl::Duration warn_stuck_timeout,
    absl::Duration terminate_timeout) {
  if (auto in_flight_rendezvous = flag.TryJoin()) {
    return Rendezvous<K>(name, key, num_threads, std::move(fn),
                         warn_stuck_timeout, terminate_timeout);
  } else {
    return std::shared_ptr<R>();
  }
}

template <typename K>
absl::Status Rendezvous(RendezvousFlag& flag, absl::string_view name,
                        const K& key, size_t num_threads,
                        absl::Duration warn_stuck_timeout,
                        absl::Duration terminate_timeout) {
  if (auto in_flight_rendezvous = flag.TryJoin()) {
    return Rendezvous<K>(name, key, num_threads, warn_stuck_timeout,
                         terminate_timeout);
  } else {
    return absl::OkStatus();
  }
}

}  // namespace xla

#endif  // XLA_SERVICE_RENDEZVOUS_H_
