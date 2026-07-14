/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_MATCHER_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_MATCHER_H_

#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace test {

// Matcher for tensorflow::Tensor instances. Two tensors match iff
//
//   - their dtypes are equal,
//   - their shapes are equal,
//   - and their contents are equal.
//
// Their contents are matched by ::testing::Pointwise() after calling .flat<T>()
// method where the type T satisfies:
//
//   ::tensorflow::DataTypeToEnum<T>::value == dtype
//
// Use this like:
//
//   EXPECT_THAT(lhs, TensorEq(rhs));
//
// All POD types and DT_STRING type tensors are supported. Note that this
// utility requires Tensors to point to CPU memory.
class TensorEq {
 public:
  explicit TensorEq(const tensorflow::Tensor& target) : target_(target) {}

  // Matchers depend on implicit casts. Do not make explicit.
  operator ::testing::Matcher<const tensorflow::Tensor&>() const;  // NOLINT

 private:
  const tensorflow::Tensor& target_;
};

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_MATCHER_H_
