/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_BASE_TEST_TENSOR_TYPES_TEST_UTIL_H_
#define TENSORFLOW_CC_EXPERIMENTAL_BASE_TEST_TENSOR_TYPES_TEST_UTIL_H_

#include <stdint.h>

#include "tensorflow/c/tf_datatype.h"

namespace tensorflow {

// Each of the following struct types have two members: a kDType that
// corresponds to a TF_Datatype enum value, and a typedef "type"
// of its corresponding C++ type. These types allow us to write Dtype-agnostic
// tests via GoogleTest's TypedTests:
// https://github.com/google/googletest/blob/e589a337170554c48bc658cc857cf15080c9eacc/googletest/docs/advanced.md#typed-tests
struct FloatType {
  using type = float;
  static constexpr TF_DataType kDType = TF_FLOAT;
};

struct DoubleType {
  using type = double;
  static constexpr TF_DataType kDType = TF_DOUBLE;
};

struct Int32Type {
  using type = int32_t;
  static constexpr TF_DataType kDType = TF_INT32;
};

struct UINT8Type {
  using type = uint8_t;
  static constexpr TF_DataType kDType = TF_UINT8;
};

struct INT8Type {
  using type = int8_t;
  static constexpr TF_DataType kDType = TF_INT8;
};

struct INT64Type {
  using type = int64_t;
  static constexpr TF_DataType kDType = TF_INT64;
};

struct UINT16Type {
  using type = uint16_t;
  static constexpr TF_DataType kDType = TF_UINT16;
};

struct UINT32Type {
  using type = uint32_t;
  static constexpr TF_DataType kDType = TF_UINT32;
};

struct UINT64Type {
  using type = uint64_t;
  static constexpr TF_DataType kDType = TF_UINT64;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_BASE_TEST_TENSOR_TYPES_TEST_UTIL_H_
