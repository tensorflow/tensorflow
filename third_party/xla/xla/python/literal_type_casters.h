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

#ifndef XLA_PYTHON_LITERAL_TYPE_CASTERS_H_
#define XLA_PYTHON_LITERAL_TYPE_CASTERS_H_

#include <cstdint>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "nanobind/nanobind.h"
#include "xla/literal.h"
#include "xla/python/types.h"
#include "xla/shape.h"

namespace nanobind {
namespace detail {

// Literals.
// Literal data can be passed to XLA as a NumPy array; its value can be
// cast to an xla::BorrowingLiteral or xla::LiteralSlice in a zero-copy way.
// We don't have any literal -> numpy conversions here, since all the methods
// that want to return arrays build Python objects directly.

template <>
struct type_caster<xla::BorrowingLiteral> {
 public:
  using Value = xla::BorrowingLiteral;
  static constexpr auto Name = const_name("xla::BorrowingLiteral");  // NOLINT
  template <typename T_>
  using Cast = movable_cast_t<T_>;
  explicit operator Value*() { return &value; }
  explicit operator Value&() { return (Value&)value; }
  explicit operator Value&&() { return (Value&&)value; }
  Value value;

  // Pybind appears to keep type_casters alive until the callee has run.
  absl::InlinedVector<nanobind::object, 1> arrays;

  bool from_python(handle input, uint8_t, cleanup_list*) noexcept {
    // TODO(b/79707221): support nested tuples if/when XLA adds support for
    // nested BorrowingLiterals.
    if (nanobind::isinstance<nanobind::tuple>(input)) {
      nanobind::tuple tuple = nanobind::borrow<nanobind::tuple>(input);
      std::vector<xla::Shape> shapes;
      std::vector<const char*> buffers;
      arrays.reserve(tuple.size());
      shapes.reserve(tuple.size());
      buffers.reserve(tuple.size());
      for (nanobind::handle entry : tuple) {
        auto c = xla::CastToArray(entry);
        if (!c) {
          return false;
        }
        arrays.push_back(c->array);
        buffers.push_back(c->buf_ptr);
        shapes.push_back(c->shape);
      }
      value = xla::BorrowingLiteral(buffers,
                                    xla::ShapeUtil::MakeTupleShape(shapes));
    } else {
      auto c = xla::CastToArray(input);
      if (!c) {
        return false;
      }
      arrays.push_back(c->array);
      value = xla::BorrowingLiteral(c->buf_ptr, c->shape);
    }
    return true;
  }
};

template <>
struct type_caster<xla::LiteralSlice> {
 public:
  NB_TYPE_CASTER(xla::LiteralSlice, const_name("xla::LiteralSlice"));

  // Pybind appears to keep type_casters alive until the callee has run.
  type_caster<xla::BorrowingLiteral> literal_caster;

  bool from_python(handle handle, uint8_t flags,
                   cleanup_list* cleanup) noexcept {
    if (!literal_caster.from_python(handle, flags, cleanup)) {
      return false;
    }
    value = static_cast<const xla::BorrowingLiteral&>(literal_caster);
    return true;
  }

  static handle from_cpp(xla::LiteralSlice src, rv_policy policy,
                         cleanup_list* cleanup) noexcept {
    PyErr_Format(PyExc_NotImplementedError,
                 "LiteralSlice::from_cpp not implemented");
    return handle();
  }
};

}  // namespace detail
}  // namespace nanobind

#endif  // XLA_PYTHON_LITERAL_TYPE_CASTERS_H_
