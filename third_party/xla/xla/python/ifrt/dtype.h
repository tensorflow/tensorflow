/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_DTYPE_H_
#define XLA_PYTHON_IFRT_DTYPE_H_

#include <optional>
#include <ostream>
#include <string>

#include "absl/status/statusor.h"
#include "xla/python/ifrt/dtype.pb.h"

namespace xla {
namespace ifrt {

// Data type of an element.
//
// Based on `xla::PrimitiveType`. Differences:
//
// * Match the Google C++ style guide for enumerator naming.
// * Rename PRIMITIVE_TYPE_INVALID to kInvalid.
// * Remove TUPLE, OPAQUE_TYPE.
// * Add kString.
class DType {
 public:
  // LINT.IfChange
  enum Kind {
    // Invalid data type.
    kInvalid = 0,

    // Predicates are two-state booleans.
    kPred = 1,

    // Signed integral values of fixed width.
    kS2 = 26,
    kS4 = 21,
    kS8 = 2,
    kS16 = 3,
    kS32 = 4,
    kS64 = 5,

    // Unsigned integral values of fixed width.
    kU2 = 27,
    kU4 = 22,
    kU8 = 6,
    kU16 = 7,
    kU32 = 8,
    kU64 = 9,

    // Floating-point values of fixed width.
    kF16 = 10,
    kF32 = 11,
    kF64 = 12,

    // Truncated 16 bit floating-point format. This is similar to IEEE's 16 bit
    // floating-point format, but uses 1 bit for the sign, 8 bits for the
    // exponent and 7 bits for the mantissa.
    kBF16 = 16,

    // Complex values of fixed width.
    kC64 = 15,   // Paired F32 (real, imag), as in std::complex<float>.
    kC128 = 18,  // Paired F64 (real, imag), as in std::complex<double>.

    // A token type threaded between side-effecting operations. Shapes of this
    // dtype will have empty dimensions.
    kToken = 17,

    // Opaque objects.
    kOpaque = 14,

    kF8E3M4 = 29,
    kF8E4M3 = 28,
    kF8E4M3FN = 20,
    kF8E4M3B11FNUZ = 23,
    kF8E4M3FNUZ = 25,
    kF8E5M2 = 19,
    kF8E5M2FNUZ = 24,

    // Next = 30

    // Variable-length string represented as raw bytes, as in `bytes` in Python,
    // i.e., no encoding enforcement. String is not support in XLA. DType.Kind
    // needs to match xla.PrimitiveType enum, so choose a large enum to avoid
    // collision.
    kString = 99,
  };
  // LINT.ThenChange(dtype.proto:DTypeProtoKind)

  explicit DType(Kind kind) : kind_(kind) {}
  DType(const DType&) = default;
  DType(DType&&) = default;
  DType& operator=(const DType&) = default;
  DType& operator=(DType&&) = default;

  Kind kind() const { return kind_; }

  bool operator==(const DType& other) const { return kind_ == other.kind_; }
  bool operator!=(const DType& other) const { return kind_ != other.kind_; }

  template <typename H>
  friend H AbslHashValue(H h, const DType& value) {
    return H::combine(std::move(h), value.kind());
  }

  // Returns the byte size of a single element of this DType. Returns
  // std::nullopt if not aligned to a byte boundary or there is no fixed size
  // (such as kString).
  std::optional<int> byte_size() const;

  // Returns the bit size of a single element of this DType. Returns
  // std::nullopt if there is no fixed size.
  std::optional<int> bit_size() const;

  // Constructs `DType` from `DTypeProto`.
  static absl::StatusOr<DType> FromProto(const DTypeProto& proto);

  // Returns a `DTypeProto` representation.
  DTypeProto ToProto() const;

  // TODO(hyeontaek): Remove this method in favor of AbslStringify.
  std::string DebugString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DType& dtype) {
    sink.Append(dtype.DebugString());
  }

 private:
  Kind kind_;
};

std::ostream& operator<<(std::ostream& os, const DType& dtype);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_DTYPE_H_
