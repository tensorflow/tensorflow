/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_CLEANUP_H_
#define XLA_PYTHON_IFRT_CLEANUP_H_

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace xla {
namespace ifrt {

namespace internal {

template <typename F, typename = void>
struct IsNullComparable : std::false_type {};

template <typename F>
struct IsNullComparable<
    F, std::void_t<decltype(std::declval<const F&>() == nullptr)>>
    : std::true_type {};

template <typename F>
struct SharedCleanup {
  F cleanup;

  template <typename G, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<G>, SharedCleanup>>>
  explicit SharedCleanup(G&& cleanup) : cleanup(std::forward<G>(cleanup)) {}

  SharedCleanup(const SharedCleanup&) = delete;
  SharedCleanup& operator=(const SharedCleanup&) = delete;

  ~SharedCleanup() { std::move(cleanup)(); }
};

}  // namespace internal

// Wraps a zero-argument cleanup callable in a `std::shared_ptr<const void>`
// that invokes and destroys the callable when the last reference is released.
// Returns `nullptr` if `cleanup` compares equal to `nullptr` (for example, an
// empty `std::function` or `absl::AnyInvocable`).
inline std::shared_ptr<const void> MakeSharedCleanup(std::nullptr_t) noexcept {
  return nullptr;
}

template <typename F, typename = std::enable_if_t<
                          !std::is_same_v<std::decay_t<F>, std::nullptr_t> &&
                          std::is_invocable_r_v<void, F>>>
std::shared_ptr<const void> MakeSharedCleanup(F&& cleanup) {
  if constexpr (internal::IsNullComparable<std::decay_t<F>>::value) {
    if (cleanup == nullptr) {
      return nullptr;
    }
  }
  return std::make_shared<internal::SharedCleanup<std::decay_t<F>>>(
      std::forward<F>(cleanup));
}

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_CLEANUP_H_
