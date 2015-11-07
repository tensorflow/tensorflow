#ifndef TENSORFLOW_FRAMEWORK_BFLOAT16_H_
#define TENSORFLOW_FRAMEWORK_BFLOAT16_H_

#include "tensorflow/core/platform/port.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// Compact 16-bit encoding of floating point numbers. This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.  It
// is assumed that floats are in IEEE 754 format so the representation is just
// bits 16-31 of a single precision float.
//
// NOTE: The IEEE floating point standard defines a float16 format that
// is different than this format (it has fewer bits of exponent and more
// bits of mantissa).  We don't use that format here because conversion
// to/from 32-bit floats is more complex for that format, and the
// conversion for this format is very simple.
//
// Because of the existing IEEE float16 type, we do not name our representation
// "float16" but just use "uint16".
//
// <-----our 16bits float------->
// s e e e e e e e e f f f f f f f f f f f f f f f f f f f f f f f
// <------------------------------float-------------------------->
// 3 3             2 2             1 1                           0
// 1 0             3 2             5 4                           0
//
//
// This type only supports conversion back and forth with float.
//
// This file must be compilable by nvcc.

namespace tensorflow {
struct bfloat16 {
  EIGEN_DEVICE_FUNC bfloat16() {}
  EIGEN_DEVICE_FUNC explicit bfloat16(const uint16_t v) : value(v) {}

  uint16_t value;
};

// Conversion routines between an array of float and bfloat16 of
// "size".
void FloatToBFloat16(const float* src, bfloat16* dst, int64 size);
void BFloat16ToFloat(const bfloat16* src, float* dst, int64 size);

}  // namespace tensorflow

namespace Eigen {
template <>
struct NumTraits<tensorflow::bfloat16> : GenericNumTraits<uint16_t> {};

EIGEN_STRONG_INLINE bool operator==(const tensorflow::bfloat16 a,
                                    const tensorflow::bfloat16 b) {
  return a.value == b.value;
}

}  // namespace Eigen

#endif  // TENSORFLOW_FRAMEWORK_BFLOAT16_H_
