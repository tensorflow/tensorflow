/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_
#define TENSORFLOW_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_

#include "tensorflow/stream_executor/data_type.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace stream_executor {

// Allows to represent a value that is either a host scalar or a scalar stored
// on the GPU device.
// See also the specialization for ElemT=void below.
template <typename ElemT>
class HostOrDeviceScalar {
 public:
  // Not marked as explicit because when using this constructor, we usually want
  // to set this to a compile-time constant.
  HostOrDeviceScalar(ElemT value) : value_(value), is_pointer_(false) {}
  explicit HostOrDeviceScalar(const DeviceMemory<ElemT>& pointer)
      : pointer_(pointer), is_pointer_(true) {
    CHECK_EQ(1, pointer.ElementCount());
  }

  bool is_pointer() const { return is_pointer_; }
  const DeviceMemory<ElemT>& pointer() const {
    CHECK(is_pointer());
    return pointer_;
  }
  const ElemT& value() const {
    CHECK(!is_pointer());
    return value_;
  }

 private:
  union {
    ElemT value_;
    DeviceMemory<ElemT> pointer_;
  };
  bool is_pointer_;
};

// Specialization for wrapping a dynamically-typed value (via type erasure).
template <>
class HostOrDeviceScalar<void> {
 public:
  using DataType = dnn::DataType;

  // Constructors not marked as explicit because when using this constructor, we
  // usually want to set this to a compile-time constant.

  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(float value)
      : float_(value), is_pointer_(false), dtype_(DataType::kFloat) {}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(double value)
      : double_(value), is_pointer_(false), dtype_(DataType::kDouble) {}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(Eigen::half value)
      : half_(value), is_pointer_(false), dtype_(DataType::kHalf) {}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(int8 value)
      : int8_(value), is_pointer_(false), dtype_(DataType::kInt8) {}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(int32 value)
      : int32_(value), is_pointer_(false), dtype_(DataType::kInt32) {}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(std::complex<float> value)
      : complex_float_(value),
        is_pointer_(false),
        dtype_(DataType::kComplexFloat) {}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(std::complex<double> value)
      : complex_double_(value),
        is_pointer_(false),
        dtype_(DataType::kComplexDouble) {}
  template <typename T>
  explicit HostOrDeviceScalar(const DeviceMemory<T>& pointer)
      : pointer_(pointer),
        is_pointer_(true),
        dtype_(dnn::ToDataType<T>::value) {
    CHECK_EQ(1, pointer.ElementCount());
  }
  // Construct from statically-typed version.
  template <typename T, typename std::enable_if<!std::is_same<T, void>::value,
                                                int>::type = 0>
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(const HostOrDeviceScalar<T>& other) {
    if (other.is_pointer()) {
      *this = HostOrDeviceScalar(other.pointer());
    } else {
      *this = HostOrDeviceScalar(other.value());
    }
  }

  bool is_pointer() const { return is_pointer_; }
  template <typename T>
  const DeviceMemory<T>& pointer() const {
    CHECK(is_pointer());
    CHECK(dtype_ == dnn::ToDataType<T>::value);
    return pointer_;
  }
  template <typename T>
  const T& value() const {
    CHECK(!is_pointer());
    CHECK(dtype_ == dnn::ToDataType<T>::value);
    return value_impl<T>();
  }
  const DeviceMemoryBase& opaque_pointer() const {
    CHECK(is_pointer());
    return pointer_;
  }
  const void* opaque_value() const {
    CHECK(!is_pointer());
    switch (dtype_) {
      case DataType::kFloat:
        return &float_;
      case DataType::kDouble:
        return &double_;
      case DataType::kHalf:
        return &half_;
      case DataType::kInt8:
        return &int8_;
      case DataType::kInt32:
        return &int32_;
      case DataType::kComplexFloat:
        return &complex_float_;
      case DataType::kComplexDouble:
        return &complex_double_;
      default:
        return nullptr;
    }
  }
  DataType data_type() const { return dtype_; }

 private:
  template <typename T>
  const T& value_impl() const;

  union {
    float float_;
    double double_;
    Eigen::half half_;
    int8 int8_;
    int32 int32_;
    std::complex<float> complex_float_;
    std::complex<double> complex_double_;
    DeviceMemoryBase pointer_;
  };
  bool is_pointer_;
  DataType dtype_;
};

template <>
inline const float& HostOrDeviceScalar<void>::value_impl<float>() const {
  return float_;
}

template <>
inline const double& HostOrDeviceScalar<void>::value_impl<double>() const {
  return double_;
}

template <>
inline const Eigen::half& HostOrDeviceScalar<void>::value_impl<Eigen::half>()
    const {
  return half_;
}

template <>
inline const int8& HostOrDeviceScalar<void>::value_impl<int8>() const {
  return int8_;
}

template <>
inline const int32& HostOrDeviceScalar<void>::value_impl<int32>() const {
  return int32_;
}

template <>
inline const std::complex<float>&
HostOrDeviceScalar<void>::value_impl<std::complex<float>>() const {
  return complex_float_;
}

template <>
inline const std::complex<double>&
HostOrDeviceScalar<void>::value_impl<std::complex<double>>() const {
  return complex_double_;
}

}  // namespace stream_executor
#endif  // TENSORFLOW_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_
