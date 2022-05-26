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

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/python/status_casters_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Helper that converts a failing StatusOr to an exception.
// For use only inside pybind11 code.
template <typename T>
T ValueOrThrow(StatusOr<T> v) {
  if (!v.ok()) {
    throw xla::XlaRuntimeError(v.status());
  }
  return v.ConsumeValueOrDie();
}

}  // namespace xla

// This namespace is a documented pybind11 extension point.
// Caution: Unusually for Google code, this code uses C++ exceptions because
// they are the only mechanism for reporting cast failures to pybind11. However,
// the exceptions are local to the binding code.
namespace pybind11 {
namespace detail {

// Status, StatusOr. Failing statuses become Python exceptions; Status::OK()
// becomes None.
template <>
struct type_caster<xla::Status> {
 public:
  PYBIND11_TYPE_CASTER(xla::Status, _("Status"));

  static handle cast(xla::Status src, return_value_policy /* policy */,
                     handle /* parent */) {
    if (!src.ok()) {
      std::optional<xla::status_casters_util::FunctionPtr> function =
          xla::status_casters_util::GetFunctionPointerFromPayload(src);

      if (function.has_value()) {
        function.value()(src);  // This is supposed to throw a custom exception
      }

      throw xla::XlaRuntimeError(src);
    }
    return none().inc_ref();
  }
};

template <typename T>
struct type_caster<xla::StatusOr<T>> {
 public:
  using value_conv = make_caster<T>;
  using status_conv = make_caster<xla::Status>;

  PYBIND11_TYPE_CASTER(xla::StatusOr<T>,
                       _("StatusOr[") + value_conv::name + _("]"));

  static handle cast(xla::StatusOr<T> src, return_value_policy policy,
                     handle parent) {
    if (!src.ok()) {
      return status_conv::cast(src.status(), policy, parent);
    }
    return value_conv::cast(std::forward<xla::StatusOr<T>>(src).ValueOrDie(),
                            policy, parent);
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_STATUS_CASTERS_H_
