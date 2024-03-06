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

#ifndef XLA_PYTHON_NB_HELPERS_H_
#define XLA_PYTHON_NB_HELPERS_H_

#include "third_party/nanobind/include/nanobind/nanobind.h"

namespace xla {

// Calls Python hash() on an object.
// TODO(phawkins): consider upstreaming this to nanobind.
ssize_t nb_hash(nanobind::handle o);

// Variant of NB_TYPE_CASTER that doesn't define from_cpp()
#define NB_TYPE_CASTER_FROM_PYTHON_ONLY(Value_, descr)   \
  using Value = Value_;                                  \
  static constexpr auto Name = descr;                    \
  template <typename T_>                                 \
  using Cast = movable_cast_t<T_>;                       \
  explicit operator Value*() { return &value; }          \
  explicit operator Value&() { return (Value&)value; }   \
  explicit operator Value&&() { return (Value&&)value; } \
  Value value;

}  // namespace xla

#endif  // XLA_PYTHON_NB_HELPERS_H_
