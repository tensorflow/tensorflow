/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_BYTE_SIZE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_BYTE_SIZE_H_

#include <cstddef>
#include <ostream>
#include <string>

namespace tensorflow {
namespace data {

// A `ByteSize` represents data space usage measured in bytes. It is constructed
// using Bytes, KB, MB, GB, or TB. Supports common arithmetic operations. Uses
// `size_t` in its internal representation. Thus, it only supports non-negative
// sizes, and the maximum byte size is std::numeric_limits<size_t>::max().
//
// Usage example:
//
//   constexpr ByteSize kAllocatedMemoryLimit = ByteSize::MB(64);
//
//   Tensor data = ...
//   ByteSize tensor_size = ByteSize::Bytes(data.AllocatedBytes());
//   if (tensor_size > 0.95 * kAllocatedMemoryLimit) {
//     LOG(WARNING) << "Tensor memory usage is " << tensor_size << ". This is "
//                  << "close to the limit " << kAllocatedMemoryLimit << ".";
//   }
class ByteSize final {
 public:
  // The default is 0 bytes.
  constexpr ByteSize() = default;
  constexpr ByteSize(const ByteSize&) = default;
  ByteSize& operator=(const ByteSize&) = default;

  // Constructs byte sizes of bytes, KB, MB, GB, and TB.
  constexpr static ByteSize Bytes(size_t n);

  // In this and following templates, `T` should be a numeric type,
  // e.g.: size_t, double, etc.
  template <class T>
  constexpr static ByteSize KB(T n);

  template <class T>
  constexpr static ByteSize MB(T n);

  template <class T>
  constexpr static ByteSize GB(T n);

  template <class T>
  constexpr static ByteSize TB(T n);

  // Compound assignment operators.
  ByteSize& operator+=(ByteSize rhs);

  // Does not support negative bytes. If *this < rhs, returns 0 bytes.
  ByteSize& operator-=(ByteSize rhs);

  template <class T>
  ByteSize& operator*=(T rhs);

  template <class T>
  ByteSize& operator/=(T rhs);

  // Converts the measurement into the specified unit.
  size_t ToUnsignedBytes() const;
  double ToDoubleBytes() const;
  double ToDoubleKB() const;
  double ToDoubleMB() const;
  double ToDoubleGB() const;
  double ToDoubleTB() const;

  // Returns a human-readable string of the byte size. For example, "5KB",
  // "1GB", etc.
  std::string DebugString() const;

 private:
  constexpr explicit ByteSize(double bytes) : bytes_(bytes) {}

  size_t bytes_ = 0;
};

constexpr ByteSize ByteSize::Bytes(size_t n) { return ByteSize(n); };

template <class T>
constexpr ByteSize ByteSize::KB(T n) {
  return ByteSize::Bytes(n * (size_t{1} << 10));
}

template <class T>
constexpr ByteSize ByteSize::MB(T n) {
  return ByteSize::Bytes(n * (size_t{1} << 20));
}

template <class T>
constexpr ByteSize ByteSize::GB(T n) {
  return ByteSize::Bytes(n * (size_t{1} << 30));
}

template <class T>
constexpr ByteSize ByteSize::TB(T n) {
  return ByteSize::Bytes(n * (size_t{1} << 40));
}

// Compound assignments.
inline ByteSize& ByteSize::operator+=(ByteSize rhs) {
  bytes_ += rhs.ToUnsignedBytes();
  return *this;
}

inline ByteSize& ByteSize::operator-=(ByteSize rhs) {
  if (bytes_ < rhs.ToUnsignedBytes()) {
    bytes_ = 0;
    return *this;
  }
  bytes_ -= rhs.ToUnsignedBytes();
  return *this;
}

template <class T>
inline ByteSize& ByteSize::operator*=(T rhs) {
  bytes_ *= rhs;
  return *this;
}

template <class T>
inline ByteSize& ByteSize::operator/=(T rhs) {
  bytes_ /= rhs;
  return *this;
}

// Binary arithmetic operators.
inline ByteSize operator+(ByteSize lhs, ByteSize rhs) {
  return lhs += rhs;
}

inline ByteSize operator-(ByteSize lhs, ByteSize rhs) {
  return lhs -= rhs;
}

template <class T>
inline ByteSize operator*(ByteSize lhs, T rhs) { return lhs *= rhs; }

template <class T>
inline ByteSize operator*(T lhs, ByteSize rhs) { return rhs *= lhs; }

template <class T>
inline ByteSize operator/(ByteSize lhs, T rhs) { return lhs /= rhs; }

inline double operator/(ByteSize lhs, ByteSize rhs) {
  return lhs.ToDoubleBytes() / rhs.ToDoubleBytes();
}

// Comparison operators.
inline bool operator<(ByteSize lhs, ByteSize rhs) {
  return lhs.ToUnsignedBytes() < rhs.ToUnsignedBytes();
}

inline bool operator>(ByteSize lhs, ByteSize rhs) {
  return rhs < lhs;
}

inline bool operator>=(ByteSize lhs, ByteSize rhs) {
  return !(lhs < rhs);
}

inline bool operator<=(ByteSize lhs, ByteSize rhs) {
  return !(rhs < lhs);
}

inline bool operator==(ByteSize lhs, ByteSize rhs) {
  return lhs.ToUnsignedBytes() == rhs.ToUnsignedBytes();
}

inline bool operator!=(ByteSize lhs, ByteSize rhs) {
  return !(lhs == rhs);
}

// Output operator, which supports logging with LOG(*).
inline std::ostream& operator<<(std::ostream& os, ByteSize byte_size) {
  return os << byte_size.DebugString();
}

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_BYTE_SIZE_H_
