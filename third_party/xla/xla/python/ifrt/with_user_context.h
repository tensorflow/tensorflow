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

#ifndef XLA_PYTHON_IFRT_WITH_USER_CONTEXT_H_
#define XLA_PYTHON_IFRT_WITH_USER_CONTEXT_H_

#include <functional>
#include <type_traits>
#include <utility>

#include "xla/python/ifrt/user_context.h"

namespace xla {
namespace ifrt {

namespace internal {

// Internal implementation of `WithUserContext` and `WithCurrentUserContext`.
template <typename Functor>
class WithUserContextFunctorImpl {
 public:
  explicit WithUserContextFunctorImpl(const Functor& functor,
                                      UserContextRef user_context)
      : functor_(functor), user_context_(std::move(user_context)) {}

  explicit WithUserContextFunctorImpl(Functor&& functor,
                                      UserContextRef user_context)
      : functor_(std::forward<Functor>(functor)),
        user_context_(std::move(user_context)) {}

  template <typename... Args>
  std::invoke_result_t<Functor&, Args&&...> operator()(Args&&... args) & {
    UserContextScope user_context_scope(user_context_);
    return std::invoke(functor_, std::forward<Args>(args)...);
  }

  template <typename... Args>
  std::invoke_result_t<Functor&&, Args&&...> operator()(Args&&... args) && {
    UserContextScope user_context_scope(user_context_);
    return std::invoke(std::move(functor_), std::forward<Args>(args)...);
  }

  template <typename... Args>
  std::invoke_result_t<const Functor&, Args&&...> operator()(
      Args&&... args) const& {
    UserContextScope user_context_scope(user_context_);
    return std::invoke(functor_, std::forward<Args>(args)...);
  }

  template <typename... Args>
  std::invoke_result_t<const Functor&&, Args&&...> operator()(
      Args&&... args) const&& {
    UserContextScope user_context_scope(user_context_);
    return std::invoke(std::move(functor_), std::forward<Args>(args)...);
  }

 private:
  Functor functor_;
  UserContextRef user_context_;
};

// Template alias to hide the details of the computed return type of
// `WithUserContext` and `WithCurrentUserContext`.
template <typename Functor>
using WithUserContextFunctor =
    WithUserContextFunctorImpl<std::decay_t<Functor>>;

}  // namespace internal

// Takes an invocable of some kind and returns an invocable that will set the
// given user context before invoking `invocable`.
template <typename Functor>
internal::WithUserContextFunctor<Functor> WithUserContext(
    Functor&& invocable, UserContextRef user_context) {
  return internal::WithUserContextFunctor<Functor>(
      std::forward<Functor>(invocable), std::move(user_context));
}

// Takes an invocable of some kind and returns an invocable that will set the
// current user context before invoking `invocable`.
template <typename Functor>
internal::WithUserContextFunctor<Functor> WithCurrentUserContext(
    Functor&& invocable) {
  return internal::WithUserContextFunctor<Functor>(
      std::forward<Functor>(invocable), UserContextScope::current());
}

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_WITH_USER_CONTEXT_H_
