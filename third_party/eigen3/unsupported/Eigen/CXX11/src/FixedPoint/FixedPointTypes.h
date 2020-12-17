// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CXX11_SRC_FIXEDPOINT_FIXEDPOINTTYPES_H_
#define CXX11_SRC_FIXEDPOINT_FIXEDPOINTTYPES_H_

#include <cmath>
#include <iostream>

namespace Eigen {

// The mantissa part of the fixed point representation. See
// go/tensorfixedpoint for details
struct QInt8;
struct QUInt8;
struct QInt16;
struct QUInt16;
struct QInt32;

template <>
struct NumTraits<QInt8> : GenericNumTraits<int8_t> {};
template <>
struct NumTraits<QUInt8> : GenericNumTraits<uint8_t> {};
template <>
struct NumTraits<QInt16> : GenericNumTraits<int16_t> {};
template <>
struct NumTraits<QUInt16> : GenericNumTraits<uint16_t> {};
template <>
struct NumTraits<QInt32> : GenericNumTraits<int32_t> {};

namespace internal {
template <>
struct scalar_product_traits<QInt32, double> {
  enum {
    // Cost = NumTraits<T>::MulCost,
    Defined = 1
  };
  typedef QInt32 ReturnType;
};
}

// Wrap the 8bit int into a QInt8 struct instead of using a typedef to prevent
// the compiler from silently type cast the mantissa into a bigger or a smaller
// representation.
struct QInt8 {
  QInt8() : value(0) {}
  QInt8(const int8_t v) : value(v) {}
  QInt8(const QInt32 v);

  operator int() const { return static_cast<int>(value); }

  int8_t value;
};

struct QUInt8 {
  QUInt8() : value(0) {}
  QUInt8(const uint8_t v) : value(v) {}
  QUInt8(const QInt32 v);

  operator int() const { return static_cast<int>(value); }

  uint8_t value;
};

struct QInt16 {
  QInt16() : value(0) {}
  QInt16(const int16_t v) : value(v) {}
  QInt16(const QInt32 v);
  operator int() const { return static_cast<int>(value); }

  int16_t value;
};

struct QUInt16 {
  QUInt16() : value(0) {}
  QUInt16(const uint16_t v) : value(v) {}
  QUInt16(const QInt32 v);
  operator int() const { return static_cast<int>(value); }

  uint16_t value;
};

struct QInt32 {
  QInt32() : value(0) {}
  QInt32(const int8_t v) : value(v) {}
  QInt32(const int32_t v) : value(v) {}
  QInt32(const uint32_t v) : value(static_cast<int32_t>(v)) {}
  QInt32(const QInt8 v) : value(v.value) {}
  QInt32(const float v) : value(static_cast<int32_t>(lrint(v))) {}
#ifdef EIGEN_MAKING_DOCS
  // Workaround to fix build on PPC.
  QInt32(unsigned long v) : value(v) {}
#endif

  operator float() const { return static_cast<float>(value); }

  int32_t value;
};

EIGEN_STRONG_INLINE QInt8::QInt8(const QInt32 v)
    : value(v.value > 127 ? 127 : (v.value < -128 ? -128 : v.value)) {}
EIGEN_STRONG_INLINE QUInt8::QUInt8(const QInt32 v)
    : value(v.value > 255 ? 255 : (v.value < 0 ? 0 : v.value)) {}
EIGEN_STRONG_INLINE QInt16::QInt16(const QInt32 v)
    : value(v.value > 32767 ? 32767 : (v.value < -32768 ? -32768 : v.value)) {}
EIGEN_STRONG_INLINE QUInt16::QUInt16(const QInt32 v)
    : value(v.value > 65535 ? 65535 : (v.value < 0 ? 0 : v.value)) {}

// Basic widening 8-bit operations: This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator*(const QInt8 a, const QInt8 b) {
  return QInt32(static_cast<int32_t>(a.value) * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt8 a, const QUInt8 b) {
  return QInt32(static_cast<int32_t>(a.value) * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QInt8 a, const QInt8 b) {
  return QInt32(static_cast<int32_t>(a.value) + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt8 a, const QInt8 b) {
  return QInt32(static_cast<int32_t>(a.value) - static_cast<int32_t>(b.value));
}

// Basic widening 16-bit operations: This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator*(const QInt16 a, const QInt16 b) {
  return QInt32(static_cast<int32_t>(a.value) * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt16 a, const QUInt16 b) {
  return QInt32(static_cast<int32_t>(a.value) * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QInt16 a, const QInt16 b) {
  return QInt32(static_cast<int32_t>(a.value) + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt16 a, const QInt16 b) {
  return QInt32(static_cast<int32_t>(a.value) - static_cast<int32_t>(b.value));
}

// Mixed QInt32 op QInt8 operations. This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator+(const QInt32 a, const QInt8 b) {
  return QInt32(a.value + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QInt8 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) + b.value);
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a, const QInt8 b) {
  return QInt32(a.value - static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt8 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) - b.value);
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const QInt8 b) {
  return QInt32(a.value * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt8 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) * b.value);
}

// Mixed QInt32 op QInt16 operations. This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator+(const QInt32 a, const QInt16 b) {
  return QInt32(a.value + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QInt16 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) + b.value);
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a, const QInt16 b) {
  return QInt32(a.value - static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt16 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) - b.value);
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const QInt16 b) {
  return QInt32(a.value * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt16 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) * b.value);
}

// Mixed QInt32 op QUInt8 operations. This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator+(const QInt32 a, const QUInt8 b) {
  return QInt32(a.value + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QUInt8 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) + b.value);
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a, const QUInt8 b) {
  return QInt32(a.value - static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QUInt8 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) - b.value);
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const QUInt8 b) {
  return QInt32(a.value * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QUInt8 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) * b.value);
}

// Mixed QInt32 op QUInt16 operations. This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator+(const QInt32 a, const QUInt16 b) {
  return QInt32(a.value + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QUInt16 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) + b.value);
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a, const QUInt16 b) {
  return QInt32(a.value - static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QUInt16 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) - b.value);
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const QUInt16 b) {
  return QInt32(a.value * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QUInt16 a, const QInt32 b) {
  return QInt32(static_cast<int32_t>(a.value) * b.value);
}

// Basic arithmetic operations on QInt32, which behaves like a int32_t.
EIGEN_STRONG_INLINE QInt32 operator+(const QInt32 a, const QInt32 b) {
  return a.value + b.value;
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a, const QInt32 b) {
  return a.value - b.value;
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const QInt32 b) {
  return a.value * b.value;
}
EIGEN_STRONG_INLINE QInt32 operator/(const QInt32 a, const QInt32 b) {
  return a.value / b.value;
}
EIGEN_STRONG_INLINE QInt32& operator+=(QInt32& a, const QInt32 b) {
  a.value += b.value;
  return a;
}
EIGEN_STRONG_INLINE QInt32& operator-=(QInt32& a, const QInt32 b) {
  a.value -= b.value;
  return a;
}
EIGEN_STRONG_INLINE QInt32& operator*=(QInt32& a, const QInt32 b) {
  a.value *= b.value;
  return a;
}
EIGEN_STRONG_INLINE QInt32& operator/=(QInt32& a, const QInt32 b) {
  a.value /= b.value;
  return a;
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a) { return -a.value; }

// Scaling QInt32 by double. We do the arithmetic in double because
// float only has 23 bits of mantissa, so casting QInt32 to float might reduce
// accuracy by discarding up to 7 (least significant) bits.
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const double b) {
  return static_cast<int32_t>(lrint(static_cast<double>(a.value) * b));
}
EIGEN_STRONG_INLINE QInt32 operator*(const double a, const QInt32 b) {
  return static_cast<int32_t>(lrint(a * static_cast<double>(b.value)));
}
EIGEN_STRONG_INLINE QInt32& operator*=(QInt32& a, const double b) {
  a.value = static_cast<int32_t>(lrint(static_cast<double>(a.value) * b));
  return a;
}

// Comparisons
EIGEN_STRONG_INLINE bool operator==(const QInt8 a, const QInt8 b) {
  return a.value == b.value;
}
EIGEN_STRONG_INLINE bool operator==(const QUInt8 a, const QUInt8 b) {
  return a.value == b.value;
}
EIGEN_STRONG_INLINE bool operator==(const QInt16 a, const QInt16 b) {
  return a.value == b.value;
}
EIGEN_STRONG_INLINE bool operator==(const QUInt16 a, const QUInt16 b) {
  return a.value == b.value;
}
EIGEN_STRONG_INLINE bool operator==(const QInt32 a, const QInt32 b) {
  return a.value == b.value;
}

EIGEN_STRONG_INLINE bool operator<(const QInt8 a, const QInt8 b) {
  return a.value < b.value;
}
EIGEN_STRONG_INLINE bool operator<(const QUInt8 a, const QUInt8 b) {
  return a.value < b.value;
}
EIGEN_STRONG_INLINE bool operator<(const QInt16 a, const QInt16 b) {
  return a.value < b.value;
}
EIGEN_STRONG_INLINE bool operator<(const QUInt16 a, const QUInt16 b) {
  return a.value < b.value;
}
EIGEN_STRONG_INLINE bool operator<(const QInt32 a, const QInt32 b) {
  return a.value < b.value;
}

EIGEN_STRONG_INLINE bool operator>(const QInt8 a, const QInt8 b) {
  return a.value > b.value;
}
EIGEN_STRONG_INLINE bool operator>(const QUInt8 a, const QUInt8 b) {
  return a.value > b.value;
}
EIGEN_STRONG_INLINE bool operator>(const QInt16 a, const QInt16 b) {
  return a.value > b.value;
}
EIGEN_STRONG_INLINE bool operator>(const QUInt16 a, const QUInt16 b) {
  return a.value > b.value;
}
EIGEN_STRONG_INLINE bool operator>(const QInt32 a, const QInt32 b) {
  return a.value > b.value;
}

EIGEN_STRONG_INLINE std::ostream& operator<<(std::ostream& os, QInt8 a) {
  os << static_cast<int>(a.value);
  return os;
}
EIGEN_STRONG_INLINE std::ostream& operator<<(std::ostream& os, QUInt8 a) {
  os << static_cast<int>(a.value);
  return os;
}
EIGEN_STRONG_INLINE std::ostream& operator<<(std::ostream& os, QInt16 a) {
  os << static_cast<int>(a.value);
  return os;
}
EIGEN_STRONG_INLINE std::ostream& operator<<(std::ostream& os, QUInt16 a) {
  os << static_cast<int>(a.value);
  return os;
}
EIGEN_STRONG_INLINE std::ostream& operator<<(std::ostream& os, QInt32 a) {
  os << a.value;
  return os;
}

}  // namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_FIXEDPOINTTYPES_H_
