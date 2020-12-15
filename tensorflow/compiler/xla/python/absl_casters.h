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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_ABSL_CASTERS_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_ABSL_CASTERS_H_

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace pybind11::detail {

// absl::Span
template <typename T>
struct type_caster<absl::Span<const T>> {
  using value_conv = make_caster<T>;

  PYBIND11_TYPE_CASTER(absl::Span<const T>,
                       _("Span[") + value_conv::name + _("]"));

  // absl::Span doesn't hold ownership. We therefore need a temporary array.
  // Pybind appears to keep type_casters alive until the callee has run.
  std::vector<T> storage;

  bool load(handle src, bool convert) {
    if (!isinstance<sequence>(src)) {
      return false;
    }
    auto seq = reinterpret_borrow<sequence>(src);
    storage.clear();
    storage.reserve(seq.size());
    for (const auto& it : seq) {
      value_conv conv;
      if (!conv.load(it, convert)) {
        return false;
      }
      storage.push_back(cast_op<T&&>(std::move(conv)));
    }
    value = absl::Span<const T>(storage);
    return true;
  }
};

// When absl::optional is an alias for std::optional, the type_caster
// specializations are provided by pybind11.
#ifndef ABSL_HAVE_STD_OPTIONAL
// absl::optional
template <typename T>
struct type_caster<absl::optional<T>> : optional_caster<absl::optional<T>> {};

template <>
struct type_caster<absl::nullopt_t> : public void_caster<absl::nullopt_t> {};
#endif

}  // namespace pybind11::detail

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_ABSL_CASTERS_H_
