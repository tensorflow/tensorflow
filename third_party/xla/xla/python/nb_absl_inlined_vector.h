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

#ifndef XLA_PYTHON_NB_ABSL_INLINED_VECTOR_H_
#define XLA_PYTHON_NB_ABSL_INLINED_VECTOR_H_

#include "absl/container/inlined_vector.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/detail/nb_list.h"

namespace nanobind {
namespace detail {

template <typename Type, size_t N, typename Alloc>
struct type_caster<absl::InlinedVector<Type, N, Alloc>>
    : list_caster<absl::InlinedVector<Type, N, Alloc>, Type> {};

}  // namespace detail
}  // namespace nanobind

#endif  // XLA_PYTHON_NB_ABSL_INLINED_VECTOR_H_
