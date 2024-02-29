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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "xla/literal.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

// Converts a NumPy dtype to a PrimitiveType.
absl::StatusOr<PrimitiveType> DtypeToPrimitiveType(
    const pybind11::dtype& np_type);

// Converts a PrimitiveType to a Numpy dtype.
absl::StatusOr<pybind11::dtype> PrimitiveTypeToDtype(PrimitiveType type);

// Converts an IFRT dtype to a NumPy dtype.
absl::StatusOr<pybind11::dtype> IfrtDtypeToDtype(ifrt::DType dtype);

StatusOr<ifrt::DType> DtypeToIfRtDType(pybind11::dtype dtype);

// Returns a Python buffer protocol (PEP 3118) format descriptor string for
// `type`. Return nullptr if there is no suitable choice of format string.
const char* PEP3118FormatDescriptorForPrimitiveType(PrimitiveType type);

// Returns a numpy-style typestr for `type`, as returned by np.dtype(...).str
absl::StatusOr<pybind11::str> TypeDescriptorForPrimitiveType(
    PrimitiveType type);

struct NumpyScalarTypes {
  pybind11::object np_bool;
  pybind11::object np_int4;
  pybind11::object np_int8;
  pybind11::object np_int16;
  pybind11::object np_int32;
  pybind11::object np_int64;
  pybind11::object np_uint4;
  pybind11::object np_uint8;
  pybind11::object np_uint16;
  pybind11::object np_uint32;
  pybind11::object np_uint64;
  pybind11::object np_bfloat16;
  pybind11::object np_float8_e4m3fn;
  pybind11::object np_float8_e4m3b11fnuz;
  pybind11::object np_float8_e4m3fnuz;
  pybind11::object np_float8_e5m2;
  pybind11::object np_float8_e5m2fnuz;
  pybind11::object np_float16;
  pybind11::object np_float32;
  pybind11::object np_float64;
  pybind11::object np_complex64;
  pybind11::object np_complex128;
  pybind11::object np_longlong;
  pybind11::object np_intc;
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
absl::StatusOr<pybind11::object> LiteralToPython(
    std::shared_ptr<Literal> literal);

// Converts a sequence of C++ ints to a Python tuple of ints.
// Pybind11 by default converts a std::vector<T> to a Python list;
// we frequently want a tuple instead e.g. for shapes.
template <typename T>
pybind11::tuple SpanToTuple(absl::Span<T const> xs) {
  pybind11::tuple out(xs.size());
  for (int i = 0; i < xs.size(); ++i) {
    out[i] = pybind11::cast(xs[i]);
  }
  return out;
}
template <>
pybind11::tuple SpanToTuple(absl::Span<int const> xs);
template <>
pybind11::tuple SpanToTuple(absl::Span<int64_t const> xs);

// Converts a Python iterable/sequence of T to std::vector<T>
template <typename T>
std::vector<T> IterableToVector(const pybind11::iterable& iterable) {
  std::vector<T> output;
  for (auto item : iterable) {
    output.push_back(item.cast<T>());
  }
  return output;
}
template <typename T>
std::vector<T> SequenceToVector(const pybind11::sequence& sequence) {
  std::vector<T> output;
  output.reserve(sequence.size());
  for (auto item : sequence) {
    output.push_back(item.cast<T>());
  }
  return output;
}

// Private helper function used in the implementation of the type caster for
// xla::BorrowingLiteral. Converts a Python array-like object into a buffer
// pointer and shape.
struct CastToArrayResult {
  pybind11::object array;  // Holds a reference to the array to keep it alive.
  const char* buf_ptr;
  xla::Shape shape;
};
std::optional<CastToArrayResult> CastToArray(pybind11::handle h);

}  // namespace xla

// This namespace is a documented pybind11 extension point.
// Caution: Unusually for Google code, this code uses C++ exceptions because
// they are the only mechanism for reporting cast failures to pybind11. However,
// the exceptions are local to the binding code.
namespace pybind11 {
namespace detail {

// Literals.
// Literal data can be passed to XLA as a NumPy array; its value can be
// cast to an xla::BorrowingLiteral or xla::LiteralSlice in a zero-copy way.
// We don't have any literal -> numpy conversions here, since all the methods
// that want to return arrays build Python objects directly.

template <>
struct type_caster<xla::BorrowingLiteral> {
 public:
  PYBIND11_TYPE_CASTER(xla::BorrowingLiteral, _("xla::BorrowingLiteral"));

  // Pybind appears to keep type_casters alive until the callee has run.
  absl::InlinedVector<pybind11::array, 1> arrays;

  bool load(handle input, bool) {
    // TODO(b/79707221): support nested tuples if/when XLA adds support for
    // nested BorrowingLiterals.
    if (pybind11::isinstance<pybind11::tuple>(input)) {
      pybind11::tuple tuple =
          pybind11::reinterpret_borrow<pybind11::tuple>(input);
      std::vector<xla::Shape> shapes;
      std::vector<const char*> buffers;
      arrays.reserve(tuple.size());
      shapes.reserve(tuple.size());
      buffers.reserve(tuple.size());
      for (pybind11::handle entry : tuple) {
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
  PYBIND11_TYPE_CASTER(xla::LiteralSlice, _("xla::LiteralSlice"));

  // Pybind appears to keep type_casters alive until the callee has run.
  type_caster<xla::BorrowingLiteral> literal_caster;

  bool load(handle handle, bool convert) {
    if (!literal_caster.load(handle, convert)) {
      return false;
    }
    value = static_cast<const xla::BorrowingLiteral&>(literal_caster);
    return true;
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // XLA_PYTHON_TYPES_H_
