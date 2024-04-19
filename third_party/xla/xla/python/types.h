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

#ifndef XLA_PYTHON_TYPES_H_
#define XLA_PYTHON_TYPES_H_

#include <Python.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/nb_numpy.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Converts a NumPy dtype to a PrimitiveType.
absl::StatusOr<PrimitiveType> DtypeToPrimitiveType(const nb_dtype& np_type);

// Converts a PrimitiveType to a Numpy dtype.
absl::StatusOr<nb_dtype> PrimitiveTypeToNbDtype(PrimitiveType type);

// Converts an IFRT dtype to a NumPy dtype.
absl::StatusOr<nb_dtype> IfrtDtypeToNbDtype(ifrt::DType dtype);

absl::StatusOr<ifrt::DType> DtypeToIfRtDType(nb_dtype dtype);

// Returns a Python buffer protocol (PEP 3118) format descriptor string for
// `type`. Return nullptr if there is no suitable choice of format string.
const char* PEP3118FormatDescriptorForPrimitiveType(PrimitiveType type);

// Returns a numpy-style typestr for `type`, as returned by np.dtype(...).str
absl::StatusOr<nanobind::str> TypeDescriptorForPrimitiveType(
    PrimitiveType type);

struct NumpyScalarTypes {
  nanobind::object np_bool;
  nanobind::object np_int4;
  nanobind::object np_int8;
  nanobind::object np_int16;
  nanobind::object np_int32;
  nanobind::object np_int64;
  nanobind::object np_uint4;
  nanobind::object np_uint8;
  nanobind::object np_uint16;
  nanobind::object np_uint32;
  nanobind::object np_uint64;
  nanobind::object np_bfloat16;
  nanobind::object np_float8_e4m3fn;
  nanobind::object np_float8_e4m3b11fnuz;
  nanobind::object np_float8_e4m3fnuz;
  nanobind::object np_float8_e5m2;
  nanobind::object np_float8_e5m2fnuz;
  nanobind::object np_float16;
  nanobind::object np_float32;
  nanobind::object np_float64;
  nanobind::object np_complex64;
  nanobind::object np_complex128;
  nanobind::object np_longlong;
  nanobind::object np_intc;
};
const NumpyScalarTypes& GetNumpyScalarTypes();

// For S64/U64/F64/C128 types, returns the largest 32-bit equivalent.
PrimitiveType Squash64BitTypes(PrimitiveType type);

// Returns the strides for `shape`.
std::vector<int64_t> ByteStridesForShape(const Shape& shape);
std::vector<int64_t> ByteStridesForShape(PrimitiveType element_type,
                                         absl::Span<const int64_t> dimensions,
                                         const xla::Layout& layout);
std::vector<int64_t> StridesForShape(PrimitiveType element_type,
                                     absl::Span<const int64_t> dimensions,
                                     const xla::Layout& layout);

// Converts a literal to (possibly-nested tuples of) NumPy arrays.
// The literal's leaf arrays are not copied; instead the NumPy arrays share
// buffers with the literals. Takes ownership of `literal` and keeps the
// necessary pieces alive using Python reference counting.
// Requires the GIL.
absl::StatusOr<nanobind::object> LiteralToPython(
    std::shared_ptr<Literal> literal);

template <typename T>
nanobind::tuple SpanToNbTuple(absl::Span<T const> xs) {
  nanobind::tuple out =
      nanobind::steal<nanobind::tuple>(PyTuple_New(xs.size()));
  for (int i = 0; i < xs.size(); ++i) {
    PyTuple_SET_ITEM(out.ptr(), i, nanobind::cast(xs[i]).release().ptr());
  }
  return out;
}

// Converts a sequence of Python objects to a Python tuple, stealing the
// references to the objects.
nanobind::tuple MutableSpanToNbTuple(absl::Span<nanobind::object> xs);


template <typename T>
std::vector<T> IterableToVector(const nanobind::iterable& iterable) {
  std::vector<T> output;
  for (auto item : iterable) {
    output.push_back(nanobind::cast<T>(item));
  }
  return output;
}
template <typename T>
std::vector<T> SequenceToVector(const nanobind::sequence& sequence) {
  std::vector<T> output;
  output.reserve(PySequence_Size(sequence.ptr()));
  for (auto item : sequence) {
    output.push_back(nanobind::cast<T>(item));
  }
  return output;
}

// Private helper function used in the implementation of the type caster for
// xla::BorrowingLiteral. Converts a Python array-like object into a buffer
// pointer and shape.
struct CastToArrayResult {
  nanobind::object array;  // Holds a reference to the array to keep it alive.
  const char* buf_ptr;
  xla::Shape shape;
};
std::optional<CastToArrayResult> CastToArray(nanobind::handle h);

}  // namespace xla

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

  bool from_python(handle input, uint8_t, cleanup_list*) {
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

  bool from_python(handle handle, uint8_t flags, cleanup_list* cleanup) {
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

#endif  // XLA_PYTHON_TYPES_H_
