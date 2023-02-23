/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_DTYPE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_DTYPE_H_

#include <optional>
#include <ostream>
#include <string>

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
  enum Kind {
    // Invalid data type.
    kInvalid = 0,

    // Predicates are two-state booleans.
    kPred = 1,

    // Signed integral values of fixed width.
    kS8 = 2,
    kS16 = 3,
    kS32 = 4,
    kS64 = 5,

    // Unsigned integral values of fixed width.
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

    kF8E4M3FN = 19,
    kF8E5M2 = 20,

    // Next = 21

    // String is not support in XLA. DType.Kind needs to match xla.PrimitiveType
    // enum, so choose a large enum to avoid collision.
    kString = 99,
  };

  explicit DType(Kind kind) : kind_(kind) {}
  DType(const DType&) = default;
  DType(DType&&) = default;
  DType& operator=(const DType&) = default;
  DType& operator=(DType&&) = default;

  Kind kind() const { return kind_; }

  bool operator==(const DType& other) const { return kind_ == other.kind_; }
  bool operator!=(const DType& other) const { return kind_ != other.kind_; }

  // Returns the byte size of a single element of this DType. Returns
  // std::nullopt if there is no fixed size or not aligned to a byte boundary
  // (such as kPred).
  std::optional<int> byte_size() const;

  // Returns the bit size of a single element of this DType. Returns
  // std::nullopt if there is no fixed size.
  std::optional<int> bit_size() const;

  std::string DebugString() const;

 private:
  Kind kind_;
};

std::ostream& operator<<(std::ostream& os, const DType& dtype);

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_DTYPE_H_
