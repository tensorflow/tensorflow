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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_ODML_SDPA_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_ODML_SDPA_TESTER_H_

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/builtin_op_data.h"

namespace tflite {
namespace xnnpack {

class ODMLSDPATester {
 public:
  ODMLSDPATester() = default;
  ODMLSDPATester(const ODMLSDPATester&) = delete;
  ODMLSDPATester& operator=(const ODMLSDPATester&) = delete;

  inline ODMLSDPATester& QueryShape(std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    query_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    query_size_ = ComputeSize(query_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& QueryShape() const { return query_shape_; }

  inline ODMLSDPATester& KeyShape(std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    key_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    key_size_ = ComputeSize(key_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& KeyShape() const { return key_shape_; }

  inline ODMLSDPATester& ValueShape(std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    value_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    value_size_ = ComputeSize(value_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& ValueShape() const { return value_shape_; }

  inline ODMLSDPATester& MaskShape(std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    mask_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    mask_size_ = ComputeSize(mask_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& MaskShape() const { return mask_shape_; }

  inline int32_t QuerySize() const { return query_size_; }

  inline int32_t KeySize() const { return key_size_; }

  inline int32_t ValueSize() const { return value_size_; }

  inline int32_t MaskSize() const { return mask_size_; }

  std::vector<int32_t> OutputShape() const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  std::vector<int32_t> query_shape_;
  std::vector<int32_t> key_shape_;
  std::vector<int32_t> value_shape_;
  std::vector<int32_t> mask_shape_;
  int32_t query_size_ = 1;
  int32_t key_size_ = 1;
  int32_t value_size_ = 1;
  int32_t mask_size_ = 1;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_ODML_SDPA_TESTER_H_
