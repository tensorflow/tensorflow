/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_STATUS_CASTERS_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_STATUS_CASTERS_H_

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/platform/macros.h"

namespace xla {

// C++ -> Python caster helpers.
//
// Failing statuses become Python exceptions; OK Status() becomes None.
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
inline void ThrowIfError(xla::Status src) {
  if (!src.ok()) {
    throw xla::XlaRuntimeError(src);
  }
}

// If one does not want to have to define a lambda specifying the inputs
// arguments, on can use the following wrapper.
//
// It's a fancier version, not using `std::function` equivalent to:
// template <typename StatusType, typename... Args>
// std::function<pybind11::object(Args...)> ThrowIfErrorWrapper(
//     StatusType (*f)(Args...)) {
//   return [f](Args&&... args) -> pybind11::object {
//     xla::ThrowIfError(
//         std::forward<StatusType>(f(std::forward<Args>(args)...)));
//   };
// }
// In particular, the here-above solution would not wrap lambdas.

template <typename Sig, typename F>
struct ThrowIfErrorWrapper;

// C++17 "deduction guide" that guides class template argument deduction (CTAD)
template <typename F>
ThrowIfErrorWrapper(F) -> ThrowIfErrorWrapper<decltype(&F::operator()), F>;

template <typename... Args>
ThrowIfErrorWrapper(xla::Status (&)(Args...))
    -> ThrowIfErrorWrapper<xla::Status(Args...), xla::Status (&)(Args...)>;

template <typename... Args>
struct ThrowIfErrorWrapper<xla::Status(Args...), xla::Status (&)(Args...)> {
  explicit ThrowIfErrorWrapper(xla::Status (&f)(Args...)) : func(f) {}
  void operator()(Args... args) {
    xla::ThrowIfError(func(std::forward<Args>(args)...));
  }
  xla::Status (&func)(Args...);
};
template <typename C, typename... Args, typename F>
struct ThrowIfErrorWrapper<xla::Status (C::*)(Args...), F> {
  explicit ThrowIfErrorWrapper(F&& f) : func(std::move(f)) {}
  void operator()(Args... args) {
    xla::ThrowIfError(func(std::forward<Args>(args)...));
  }
  F func;
};
template <typename C, typename... Args, typename F>
struct ThrowIfErrorWrapper<xla::Status (C::*)(Args...) const, F> {
  explicit ThrowIfErrorWrapper(F&& f) : func(std::move(f)) {}
  void operator()(Args... args) const {
    xla::ThrowIfError(func(std::forward<Args>(args)...));
  }
  F func;
};

// Utilities for `StatusOr`.
template <typename T>
T ValueOrThrow(StatusOr<T> v) {
  if (!v.ok()) {
    throw xla::XlaRuntimeError(v.status());
  }
  return std::move(v).value();
}

}  // namespace xla

// This namespace is a documented pybind11 extension point.
// Caution: Unusually for Google code, this code uses C++ exceptions because
// they are the only mechanism for reporting cast failures to pybind11. However,
// the exceptions are local to the binding code.
namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<xla::StatusOr<T>> {
 public:
  using value_conv = make_caster<T>;

  PYBIND11_TYPE_CASTER(xla::StatusOr<T>,
                       _("StatusOr[") + value_conv::name + _("]"));

  static handle cast(xla::StatusOr<T> src, return_value_policy policy,
                     handle parent) {
    if (!src.ok()) {
      throw xla::XlaRuntimeError(src.status());
    }
    return value_conv::cast(std::forward<xla::StatusOr<T>>(src).value(), policy,
                            parent);
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_STATUS_CASTERS_H_
