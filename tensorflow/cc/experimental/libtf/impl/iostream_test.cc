/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/cc/experimental/libtf/impl/none.h"
#include "tensorflow/cc/experimental/libtf/impl/scalars.h"
#include "tensorflow/cc/experimental/libtf/impl/string.h"
#include "tensorflow/cc/experimental/libtf/impl/tensor_spec.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {
namespace impl {

TEST(OStreamTest, TestInt64) {
  Int64 x(42);
  std::stringstream stream;
  stream << x;
  ASSERT_EQ(stream.str(), "42");
}

TEST(OStreamTest, TestFloat32) {
  Float32 x(0.375);  // Exactly representable as a float.
  std::stringstream stream;
  stream << x;
  ASSERT_EQ(stream.str(), "0.375");
}

TEST(OStreamTest, TestString) {
  String s("foo");
  std::stringstream stream;
  stream << s;
  ASSERT_EQ(stream.str(), "foo");
}

TEST(OStreamTest, TestNone) {
  std::stringstream stream;
  stream << None::GetInstance();
  ASSERT_EQ(stream.str(), "None");
}

TEST(OStreamTest, TestTensorSpec) {
  std::stringstream stream;
  TensorSpec tensor_spec;
  tensor_spec.shape = tensorflow::PartialTensorShape({2});
  tensor_spec.dtype = tensorflow::DT_FLOAT;
  stream << tensor_spec;
  ASSERT_EQ(stream.str(), "TensorSpec(shape = [2], dtype = 1)");
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
