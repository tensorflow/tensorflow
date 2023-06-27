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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_TRANSPOSE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_TRANSPOSE_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class TransposeTester {
 public:
  TransposeTester() = default;
  TransposeTester(const TransposeTester&) = delete;
  TransposeTester& operator=(const TransposeTester&) = delete;

  inline TransposeTester& num_dims(int32_t num_dims) {
    assert(num_dims != 0);
    this->num_dims_ = num_dims;
    return *this;
  }

  inline int32_t num_dims() const { return this->num_dims_; }

  inline TransposeTester& input_shape(std::vector<int32_t> input_shape) {
    this->input_shape_ = input_shape;
    return *this;
  }

  inline const std::vector<int32_t>& input_shape() const {
    return this->input_shape_;
  }

  inline TransposeTester& perm(std::vector<int32_t> perm) {
    this->perm_ = perm;
    return *this;
  }

  inline const std::vector<int32_t>& perm() const { return this->perm_; }

  template <class T>
  void Test(TensorType tensor_type, Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TensorType tensor_type, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(TensorType tensor_type) const;

  int32_t num_dims_ = 1;
  std::vector<int32_t> input_shape_;
  std::vector<int32_t> perm_;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_TRANSPOSE_TESTER_H_
