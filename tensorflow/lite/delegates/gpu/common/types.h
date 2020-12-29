/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TYPES_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TYPES_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include <fp16.h>

namespace tflite {
namespace gpu {

// TODO(akulik): make these types Google-style compliant.

using HalfBits = uint16_t;

class alignas(2) half {
 public:
  HalfBits bits;

  half() = default;

  half(const half& f) : bits(f.bits) {}

  explicit half(float other) { bits = fp16_ieee_from_fp32_value(other); }

  void operator=(float f) { *this = half(f); }

  operator float() const { return fp16_ieee_to_fp32_value(bits); }
};

template <typename T>
struct alignas(sizeof(T)) Vec4 {
  union {
    struct {
      T x, y, z, w;
    };
    std::array<T, 4> data_;
  };

  Vec4() : Vec4(T(0.0f)) {}

  template <typename S>
  Vec4(S x_, S y_, S z_, S w_) : x(x_), y(y_), z(z_), w(w_) {}
  explicit Vec4(T v) : x(v), y(v), z(v), w(v) {}

  template <typename S>
  explicit Vec4(S v) : x(v), y(v), z(v), w(v) {}

  Vec4(const Vec4& f) : x(f.x), y(f.y), z(f.z), w(f.w) {}

  template <typename S>
  Vec4(const Vec4<S>& f) : x(f.x), y(f.y), z(f.z), w(f.w) {}

  Vec4& operator=(const Vec4& other) {
    x = other.x;
    y = other.y;
    z = other.z;
    w = other.w;
    return *this;
  }

  static constexpr int size() { return 4; }

  T& operator[](size_t n) { return data_[n]; }
  T operator[](size_t n) const { return data_[n]; }

  bool operator==(const Vec4& value) const {
    return data_[0] == value[0] && data_[1] == value[1] &&
           data_[2] == value[2] && data_[3] == value[3];
  }
  bool operator!=(const Vec4& value) const {
    return !(this->operator==(value));
  }
};

template <typename T>
struct alignas(sizeof(T)) Vec3 {
  union {
    struct {
      T x, y, z;
    };
    std::array<T, 3> data_;
  };

  Vec3() : Vec3(T(0.0f)) {}

  template <typename S>
  constexpr Vec3(S x_, S y_, S z_) : x(x_), y(y_), z(z_) {}
  explicit Vec3(T v) : x(v), y(v), z(v) {}

  template <typename S>
  explicit Vec3(S v) : x(v), y(v), z(v) {}

  Vec3(const Vec3& f) : x(f.x), y(f.y), z(f.z) {}

  template <typename S>
  Vec3(const Vec3<S>& f) : x(f.x), y(f.y), z(f.z) {}

  Vec3& operator=(const Vec3& other) {
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  static constexpr int size() { return 3; }

  T& operator[](size_t n) { return data_[n]; }
  T operator[](size_t n) const { return data_[n]; }
  bool operator==(const Vec3& value) const {
    return data_[0] == value[0] && data_[1] == value[1] && data_[2] == value[2];
  }
  bool operator!=(const Vec3& value) const {
    return !(this->operator==(value));
  }
};

template <typename T>
struct alignas(sizeof(T)) Vec2 {
  union {
    struct {
      T x, y;
    };
    std::array<T, 2> data_;
  };

  Vec2() : Vec2(T(0.0f)) {}

  template <typename S>
  Vec2(S x_, S y_) : x(x_), y(y_) {}
  explicit Vec2(T v) : x(v), y(v) {}

  template <typename S>
  explicit Vec2(S v) : x(v), y(v) {}

  Vec2(const Vec2& f) : x(f.x), y(f.y) {}

  template <typename S>
  Vec2(const Vec2<S>& f) : x(f.x), y(f.y) {}

  Vec2& operator=(const Vec2& other) {
    x = other.x;
    y = other.y;
    return *this;
  }

  bool operator==(const Vec2& value) const {
    return data_[0] == value[0] && data_[1] == value[1];
  }

  bool operator!=(const Vec2& value) const {
    return !(this->operator==(value));
  }

  static constexpr int size() { return 2; }

  T& operator[](size_t n) { return data_[n]; }
  T operator[](size_t n) const { return data_[n]; }
};

using float2 = Vec2<float>;
using half2 = Vec2<half>;
using byte2 = Vec2<int8_t>;
using ubyte2 = Vec2<uint8_t>;
using short2 = Vec2<int16_t>;
using ushort2 = Vec2<uint16_t>;
using int2 = Vec2<int32_t>;
using uint2 = Vec2<uint32_t>;

using float3 = Vec3<float>;
using half3 = Vec3<half>;
using byte3 = Vec3<int8_t>;
using ubyte3 = Vec3<uint8_t>;
using short3 = Vec3<int16_t>;
using ushort3 = Vec3<uint16_t>;
using int3 = Vec3<int32_t>;
using uint3 = Vec3<uint32_t>;

using float4 = Vec4<float>;
using half4 = Vec4<half>;
using byte4 = Vec4<int8_t>;
using ubyte4 = Vec4<uint8_t>;
using short4 = Vec4<int16_t>;
using ushort4 = Vec4<uint16_t>;
using int4 = Vec4<int32_t>;
using uint4 = Vec4<uint32_t>;

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TYPES_H_
