/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_NB_ABSL_SPAN_H_
#define XLA_PYTHON_NB_ABSL_SPAN_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/detail/nb_list.h"
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep

namespace nanobind {
namespace detail {

template <typename T>
struct type_caster<absl::Span<T const>> {
  NB_TYPE_CASTER(absl::Span<T const>,
                 const_name("Span[") + make_caster<T>::Name + const_name("]"))

  using Caster = make_caster<T>;

  list_caster<std::vector<T>, T> vec_caster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (!vec_caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = vec_caster.value;
    return true;
  }

  static handle from_cpp(absl::Span<T const> src, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    object ret = steal(PyList_New(src.size()));
    if (ret.is_valid()) {
      Py_ssize_t i = 0;
      for (const T &value : src) {
        handle h = Caster::from_cpp(value, policy, cleanup);
        if (!h.is_valid()) {
          ret.reset();
          break;
        }
        PyList_SET_ITEM(ret.ptr(), i++, h.ptr());
      }
    }
    return ret.release();
  }
};

}  // namespace detail
}  // namespace nanobind

#endif  // XLA_PYTHON_NB_ABSL_SPAN_H_
