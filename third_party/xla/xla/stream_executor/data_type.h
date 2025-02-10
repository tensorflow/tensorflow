/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_DATA_TYPE_H_
#define XLA_STREAM_EXECUTOR_DATA_TYPE_H_

#include <complex>
#include <cstdint>

#include "xla/tsl/protobuf/dnn.pb.h"
#include "tsl/platform/ml_dtypes.h"

namespace Eigen {
struct bfloat16;
struct half;
}  // namespace Eigen

namespace stream_executor {
namespace dnn {

// A helper class to convert C/C++ types to the proper enums.
template <typename T>
struct ToDataType;

// Note: If you add a new specialization below, make sure to add the
// corresponding definition in stream_executor/dnn.cc.
template <>
struct ToDataType<tsl::float4_e2m1fn> {
  static constexpr DataType value = DataType::kF4E2M1FN;
};
template <>
struct ToDataType<tsl::float8_e3m4> {
  static constexpr DataType value = DataType::kF8E3M4;
};
template <>
struct ToDataType<tsl::float8_e4m3> {
  static constexpr DataType value = DataType::kF8E4M3;
};
template <>
struct ToDataType<tsl::float8_e4m3fn> {
  static constexpr DataType value = DataType::kF8E4M3FN;
};
template <>
struct ToDataType<tsl::float8_e5m2> {
  static constexpr DataType value = DataType::kF8E5M2;
};
template <>
struct ToDataType<tsl::float8_e4m3fnuz> {
  static constexpr DataType value = DataType::kF8E4M3FNUZ;
};
template <>
struct ToDataType<tsl::float8_e5m2fnuz> {
  static constexpr DataType value = DataType::kF8E5M2FNUZ;
};
template <>
struct ToDataType<tsl::float8_e8m0fnu> {
  static constexpr DataType value = DataType::kF8E8M0FNU;
};
template <>
struct ToDataType<float> {
  static constexpr DataType value = DataType::kFloat;
};
template <>
struct ToDataType<double> {
  static constexpr DataType value = DataType::kDouble;
};
template <>
struct ToDataType<Eigen::half> {
  static constexpr DataType value = DataType::kHalf;
};
template <>
struct ToDataType<Eigen::bfloat16> {
  static constexpr DataType value = DataType::kBF16;
};
template <>
struct ToDataType<int8_t> {
  static constexpr DataType value = DataType::kInt8;
};
template <>
struct ToDataType<int32_t> {
  static constexpr DataType value = DataType::kInt32;
};
template <>
struct ToDataType<int64_t> {
  static constexpr DataType value = DataType::kInt64;
};
template <>
struct ToDataType<std::complex<float>> {
  static constexpr DataType value = DataType::kComplexFloat;
};
template <>
struct ToDataType<std::complex<double>> {
  static constexpr DataType value = DataType::kComplexDouble;
};

}  // namespace dnn
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DATA_TYPE_H_
