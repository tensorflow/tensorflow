/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_UTILS_CONCURRENCY_CONCURRENCY_UTILS_H_
#define XLA_HLO_UTILS_CONCURRENCY_CONCURRENCY_UTILS_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tsl/concurrency/executor.h"

namespace xla::concurrency {

// Returns default per-process executor that can be used in HLO passes for
// parallelizing HLO computation processing. Thread pool size is inferred
// automatically based on the number of available CPUs on the underlying
// machine.
tsl::Executor& DefaultExecutor();

// Dispatches `Action` for all items in [begin, end) range on the given
// executor. Skips executing actions after encountering first error.
template <typename T, typename ForwardIt, typename Action>
std::vector<Future<T>> Dispatch(ForwardIt begin, ForwardIt end, Action& action,
                                tsl::Executor& executor);

// Available in C++20 as std::iter_value_t.
template <typename It>
using iter_value_t = typename std::iterator_traits<It>::value_type;  // NOLINT

// Runs an action on all elements from an iterator. A successful run collects
// all the return values from actions. The implementation guarantees that the
// order of returned values corresponds to the order of elements in the argument
// iterator [action(begin), ... action(end-1)]. Note that the action can mutate
// the objects it receives from the iterator according to their semantics.
//
// The overload below is for actions that return a value. `T` must be default
// constructible.
//
// Returns synchronously when all actions finish. Aborts the run on the first
// failure. If a run aborts the underlying data is likely to be corrupted or
// partially modified.
//
// For synchronization, clients should make sure that actions do not deadlock or
// corrupt any state they access. Specifically, if actions access any shared
// mutable state clients must make sure that such access is synchronized.  The
// run can deadlock in all the standard ways. Specifically, if the action locks
// a set of shared resources make sure that all locks are acquired in the same
// order.
template <typename T, typename ForwardIt,
          std::enable_if_t<!std::is_void_v<T>>* = nullptr>
absl::StatusOr<std::vector<T>> ForEach(
    ForwardIt begin, ForwardIt end,
    absl::AnyInvocable<absl::StatusOr<T>(iter_value_t<ForwardIt> arg)> action,
    tsl::Executor& executor) {
  auto futures = Dispatch<T>(begin, end, action, executor);
  return JoinFutures<T>(absl::MakeSpan(futures)).Await();
}

// Runs an action on all elements from an iterator. Note that the action must be
// side-effecting to make any sense, and specifically it can be mutating.
//
// Returns synchronously when all actions finish. Aborts the run on the first
// failure. If a run aborts the underlying data is likely to be corrupted or
// partially modified.
//
// For synchronization, clients should make sure that actions do not deadlock or
// corrupt any state they access. Specifically, if actions access any shared
// mutable state clients must make sure that such access is synchronized.  The
// run can deadlock in all the standard ways. Specifically, if the action locks
// a set of shared resources make sure that all locks are acquired in the same
// order.
template <typename ForwardIt>
absl::Status ForEach(
    ForwardIt begin, ForwardIt end,
    absl::AnyInvocable<absl::Status(iter_value_t<ForwardIt> arg)> action,
    tsl::Executor& executor) {
  auto futures = Dispatch<void>(begin, end, action, executor);
  return JoinFutures(absl::MakeSpan(futures)).Await();
}

// Specializes `ForEach` for an iterator of `xla::HloComputation` and provides a
// parameter to use when combining return values from individual actions.
template <typename R, typename T, typename ForwardIt>
absl::StatusOr<R> ForEachHloComputation(
    ForwardIt begin, ForwardIt end,
    absl::AnyInvocable<absl::StatusOr<T>(HloComputation*)> action,
    absl::AnyInvocable<absl::StatusOr<R>(std::vector<T>&)> combiner,
    tsl::Executor& executor) {
  auto result_for_each = ForEach(begin, end, std::move(action), executor);
  if (!result_for_each.ok()) {
    return result_for_each.status();
  }
  return combiner(*result_for_each);
}

// Specializes `ForEach` for a span of `xla::HloComputation` and provides a
// parameter to use when combining return values from individual actions.
template <typename R, typename T>
absl::StatusOr<R> ForEachHloComputation(
    absl::Span<HloComputation* const> computations,
    absl::AnyInvocable<absl::StatusOr<T>(HloComputation*)> action,
    absl::AnyInvocable<absl::StatusOr<R>(std::vector<T>&)> combiner,
    tsl::Executor& executor) {
  return ForEachHloComputation(computations.begin(), computations.end(),
                               std::move(action), std::move(combiner),
                               executor);
}

// Specializes `ForEachHloComputation` to take an `xla::HloModule` and run on
// all computations in it.
template <typename R, typename T>
absl::StatusOr<R> ForEachHloComputation(
    HloModule* module,
    absl::AnyInvocable<absl::StatusOr<T>(HloComputation*)> action,
    absl::AnyInvocable<absl::StatusOr<R>(std::vector<T>&)> combiner,
    tsl::Executor& executor) {
  // The returned type is not a `forward_iterator` so we create one.
  auto it = module->computations();
  std::vector<HloComputation*> computations{it.begin(), it.end()};
  return ForEachHloComputation(computations, std::move(action),
                               std::move(combiner), executor);
}

// Specializes `ForEachHloComputation` to take an `xla::HloModule` and run on
// all non-fusion computations in it.
template <typename R, typename T>
absl::StatusOr<R> ForEachNonfusionHloComputation(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::AnyInvocable<absl::StatusOr<T>(HloComputation*)> action,
    absl::AnyInvocable<absl::StatusOr<R>(std::vector<T>&)> combiner,
    tsl::Executor& executor) {
  auto computations = module->MakeNonfusionComputations(execution_threads);
  return ForEachHloComputation(computations, std::move(action),
                               std::move(combiner), executor);
}

//===----------------------------------------------------------------------===//
// Dispatch implementation detail.
//===----------------------------------------------------------------------===//

// Dispatches `Action` for all items in [begin, end) range on the given
// executor. Skips executing actions after encountering first error.
template <typename T, typename ForwardIt, typename Action>
std::vector<Future<T>> Dispatch(ForwardIt begin, ForwardIt end, Action& action,
                                tsl::Executor& executor) {
  // Check if the iterator category is a base class of std::forward_iterator_tag
  using Category = typename std::iterator_traits<ForwardIt>::iterator_category;
  static_assert(std::is_base_of<std::forward_iterator_tag, Category>::value,
                "Iterator must be a forward iterator or stronger");

  // Deduce absl::Status or absl::StatusOr action result type.
  using R = std::invoke_result_t<Action, decltype(*begin)>;

  size_t n = std::distance(begin, end);
  std::vector<Future<T>> futures(n);

  // Abort flag for early termination of pending tasks on first error.
  auto lock = std::make_shared<absl::Mutex>();
  auto first_error = std::make_shared<absl::Status>();

  // Launch actions on the underlying executor.
  size_t i = 0;
  for (ForwardIt it = begin; it != end; ++it, ++i) {
    futures[i] = MakeFutureOn<T>(
        executor, [&, lock, first_error, argument = *it]() -> R {
          // Short-circuit if execution of `ForEach` already failed.
          {
            absl::MutexLock ml(*lock);
            if (!first_error->ok()) {
              return *first_error;
            }
          }

          R result = action(argument);
          if (!result.ok()) {
            absl::MutexLock ml(*lock);
            if (first_error->ok()) {
              if constexpr (std::is_same_v<R, absl::Status>) {
                *first_error = result;
              } else if constexpr (std::is_same_v<R, absl::StatusOr<T>>) {
                *first_error = result.status();
              }
            }
          }

          return result;
        });
  }

  return futures;
}

}  // namespace xla::concurrency

#endif  // XLA_HLO_UTILS_CONCURRENCY_CONCURRENCY_UTILS_H_
