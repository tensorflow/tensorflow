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

#ifndef XLA_HLO_UTILS_CONCURRENCY_TYPE_ADAPTERS_H_
#define XLA_HLO_UTILS_CONCURRENCY_TYPE_ADAPTERS_H_

#include <memory>
#include <vector>

#include "absl/functional/any_invocable.h"

namespace xla::concurrency {

// Turn a move-only & call-once function to copyable by caching.
//
// Basically a `absl::AnyInvocable<R(T)&&>` -> `std::function<R(T)>`.
template <typename R>
class TurnMoveOnlyToCopyableWithCaching {
 public:
  using InnerFunT = absl::AnyInvocable<R() &&>;
  explicit TurnMoveOnlyToCopyableWithCaching(InnerFunT inner_fun)
      : fun_{std::make_shared<InnerFunT>(std::move(inner_fun))} {}

  // Wraps each element of a vector of move-only functions to make them
  // copyable.
  static std::vector<TurnMoveOnlyToCopyableWithCaching<R>> FromVector(
      std::vector<InnerFunT> funs) {
    std::vector<TurnMoveOnlyToCopyableWithCaching<R>> res;
    res.reserve(funs.size());
    for (auto& f : funs) {
      res.emplace_back(std::move(f));
    }
    return res;
  }

  // Make it callable.
  R operator()() {
    if (res_ == nullptr) {
      res_ = std::make_shared<R>(std::move(*fun_)());
    }
    return *res_;
  }

 private:
  std::shared_ptr<InnerFunT> fun_ = nullptr;
  std::shared_ptr<R> res_ = nullptr;
};

// CADT
template <typename R>
TurnMoveOnlyToCopyableWithCaching(R r) -> TurnMoveOnlyToCopyableWithCaching<R>;

}  // namespace xla::concurrency
#endif  // XLA_HLO_UTILS_CONCURRENCY_TYPE_ADAPTERS_H_
