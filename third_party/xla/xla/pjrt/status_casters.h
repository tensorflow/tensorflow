/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_PJRT_STATUS_CASTERS_H_
#define XLA_PJRT_STATUS_CASTERS_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/exceptions.h"
#include "tsl/platform/macros.h"

namespace xla {

// C++ -> Python caster helpers.
//
// Failing statuses become Python exceptions; OK absl::Status() becomes None.
//
// Given there can be only a single global pybind11 type_caster for the
// `absl::Status` type, and given XLA wants a custom exception being raised,
// we use a dedicated helper to implement this feature without relying on a
// global `type_caster`.
//
// For example:
//
// - Functions without arguments:
//   m.def("my_func", []() { xla::ThrowIfError(MyFunc()); }
// - Classes with a single argument:
//   py_class.def("delete", [](Buffer& self) {
//     xla::ThrowIfError(self.Delete());
//   }
//
// For functions with more arguments, you can either inline the arguments,
// or use the `ThrowIfErrorWrapper` wrapper defined below:
//
// m.def("my_func", xla::ThrowIfErrorWrapper(MyFunc));
//
// Nonstatic member functions can be wrapped by passing a
// pointer-to-member-function:
// xla::ThrowIfErrorWrapper(&MyClass::MyMethod)

inline void ThrowIfError(absl::Status src) {
  if (!src.ok()) {
    throw xla::XlaRuntimeError(src);
  }
}

// If one does not want to have to define a lambda specifying the inputs
// arguments, on can use the `ThrowIfErrorWrapper` wrapper.
//
// There are three specializations:
// - For free functions, `Sig` is the function type and `F` is `Sig&`.
// - For callable types, `Sig` is the pointer to member function type
//   and `F` is the type of the callable.
// - For a nonstatic member function of a class `C`, `Sig` is the function type
//   and `F` is Sig C::*.
//
// In the first two cases, the wrapper returns a callable with signature `Sig`;
// in the third case, the wrapper returns callable with a modified signature
// that takes a C instance as the first argument.
template <typename Sig, typename F>
struct ThrowIfErrorWrapper;

// C++17 "deduction guide" that guides class template argument deduction (CTAD)
// For free functions.
template <typename F>
ThrowIfErrorWrapper(F) -> ThrowIfErrorWrapper<decltype(&F::operator()), F>;

// For callable types (with operator()).
template <typename... Args>
ThrowIfErrorWrapper(absl::Status (&)(Args...))
    -> ThrowIfErrorWrapper<absl::Status(Args...), absl::Status (&)(Args...)>;

// For unbound nonstatic member functions.
template <typename C, typename... Args>
ThrowIfErrorWrapper(absl::Status (C::*)(Args...))
    -> ThrowIfErrorWrapper<absl::Status(Args...), C>;

// Template specializations.

// For free functions.
template <typename... Args>
struct ThrowIfErrorWrapper<absl::Status(Args...), absl::Status (&)(Args...)> {
  explicit ThrowIfErrorWrapper(absl::Status (&f)(Args...)) : func(f) {}
  void operator()(Args... args) const {
    xla::ThrowIfError(func(std::forward<Args>(args)...));
  }
  absl::Status (&func)(Args...);
};

// For callable types (with operator()), non-const and const versions.
template <typename C, typename... Args, typename F>
struct ThrowIfErrorWrapper<absl::Status (C::*)(Args...), F> {
  explicit ThrowIfErrorWrapper(F&& f) : func(std::move(f)) {}
  void operator()(Args... args) const {
    xla::ThrowIfError(func(std::forward<Args>(args)...));
  }
  F func;
};
template <typename C, typename... Args, typename F>
struct ThrowIfErrorWrapper<absl::Status (C::*)(Args...) const, F> {
  explicit ThrowIfErrorWrapper(F&& f) : func(std::move(f)) {}
  void operator()(Args... args) const {
    xla::ThrowIfError(func(std::forward<Args>(args)...));
  }
  F func;
};

// For unbound nonstatic member functions, non-const and const versions.
// `ptmf` stands for "pointer to member function".
template <typename C, typename... Args>
struct ThrowIfErrorWrapper<absl::Status(Args...), C> {
  explicit ThrowIfErrorWrapper(absl::Status (C::*ptmf)(Args...)) : ptmf(ptmf) {}
  void operator()(C& instance, Args... args) const {
    xla::ThrowIfError((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::Status (C::*ptmf)(Args...);
};
template <typename C, typename... Args>
struct ThrowIfErrorWrapper<absl::Status(Args...) const, C> {
  explicit ThrowIfErrorWrapper(absl::Status (C::*ptmf)(Args...) const)
      : ptmf(ptmf) {}
  void operator()(const C& instance, Args... args) const {
    xla::ThrowIfError((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::Status (C::*ptmf)(Args...) const;
};

// Utilities for `StatusOr`.
template <typename T>
T ValueOrThrow(absl::StatusOr<T> v) {
  if (!v.ok()) {
    throw xla::XlaRuntimeError(v.status());
  }
  return std::move(v).value();
}

template <typename Sig, typename F>
struct ValueOrThrowWrapper;

template <typename F>
ValueOrThrowWrapper(F) -> ValueOrThrowWrapper<decltype(&F::operator()), F>;

template <typename R, typename... Args>
ValueOrThrowWrapper(absl::StatusOr<R> (&)(Args...))
    -> ValueOrThrowWrapper<absl::StatusOr<R>(Args...),
                           absl::StatusOr<R> (&)(Args...)>;

template <typename C, typename R, typename... Args>
ValueOrThrowWrapper(absl::StatusOr<R> (C::*)(Args...))
    -> ValueOrThrowWrapper<absl::StatusOr<R>(Args...), C>;

// Deduction guide for const methods.
template <typename C, typename R, typename... Args>
ValueOrThrowWrapper(absl::StatusOr<R> (C::*)(Args...) const)
    -> ValueOrThrowWrapper<absl::StatusOr<R>(Args...) const, C>;

template <typename R, typename... Args>
struct ValueOrThrowWrapper<absl::StatusOr<R>(Args...),
                           absl::StatusOr<R> (&)(Args...)> {
  explicit ValueOrThrowWrapper(absl::StatusOr<R> (&f)(Args...)) : func(f) {}
  R operator()(Args... args) const {
    return xla::ValueOrThrow(func(std::forward<Args>(args)...));
  }
  absl::StatusOr<R> (&func)(Args...);
};
template <typename R, typename C, typename... Args, typename F>
struct ValueOrThrowWrapper<absl::StatusOr<R> (C::*)(Args...), F> {
  explicit ValueOrThrowWrapper(F&& f) : func(std::move(f)) {}
  R operator()(Args... args) const {
    return xla::ValueOrThrow(func(std::forward<Args>(args)...));
  }
  F func;
};
template <typename R, typename C, typename... Args, typename F>
struct ValueOrThrowWrapper<absl::StatusOr<R> (C::*)(Args...) const, F> {
  explicit ValueOrThrowWrapper(F&& f) : func(std::move(f)) {}
  R operator()(Args... args) const {
    return xla::ValueOrThrow(func(std::forward<Args>(args)...));
  }
  F func;
};

// For unbound nonstatic member functions, non-const and const versions.
// `ptmf` stands for "pointer to member function".
template <typename R, typename C, typename... Args>
struct ValueOrThrowWrapper<absl::StatusOr<R>(Args...), C> {
  explicit ValueOrThrowWrapper(absl::StatusOr<R> (C::*ptmf)(Args...))
      : ptmf(ptmf) {}
  R operator()(C& instance, Args... args) const {
    return xla::ValueOrThrow((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::StatusOr<R> (C::*ptmf)(Args...);
};
template <typename R, typename C, typename... Args>
struct ValueOrThrowWrapper<absl::StatusOr<R>(Args...) const, C> {
  explicit ValueOrThrowWrapper(absl::StatusOr<R> (C::*ptmf)(Args...) const)
      : ptmf(ptmf) {}
  R operator()(const C& instance, Args... args) const {
    return xla::ValueOrThrow((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::StatusOr<R> (C::*ptmf)(Args...) const;
};

}  // namespace xla

#endif  // XLA_PJRT_STATUS_CASTERS_H_
