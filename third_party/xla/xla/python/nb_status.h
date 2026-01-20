/* Copyright 2025 Google LLC

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

#ifndef XLA_PYTHON_NB_STATUS_H_
#define XLA_PYTHON_NB_STATUS_H_

#include <cstdint>

#include "absl/status/status.h"
#include "nanobind/nanobind.h"

namespace nanobind {

namespace detail {

// A Nanobind type caster for absl::Status. This type caster allows Nanobind to
// automatically convert a non-OK return status to a Python exception.
template <>
struct type_caster<absl::Status> {
  NB_TYPE_CASTER(absl::Status, const_name("Status"))

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    value = absl::OkStatus();
    return true;
  }

  template <typename T>
  static handle from_cpp(T &&value, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    if (!value.ok()) {
      PyErr_Format(PyExc_RuntimeError, "absl::Status not ok: %s",
                   value.ToString().c_str());
      return nullptr;
    } else {
      return none().release();
    }
  }
};

}  // namespace detail
}  // namespace nanobind

#endif  // XLA_PYTHON_NB_STATUS_H_
