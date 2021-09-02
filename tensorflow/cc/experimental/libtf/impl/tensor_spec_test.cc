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

#include "tensorflow/cc/experimental/libtf/impl/tensor_spec.h"

#include "absl/hash/hash_testing.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {
namespace impl {

TEST(TensorSpecTest, TestSupportsAbslHash) {
  tensorflow::PartialTensorShape unknown_shape;
  TensorSpec ts1;
  ts1.shape = unknown_shape;
  ts1.dtype = tensorflow::DT_FLOAT;

  TensorSpec ts2;
  ts2.shape = tensorflow::PartialTensorShape({2});
  ts2.dtype = tensorflow::DT_FLOAT;

  TensorSpec ts3;
  ts3.shape = tensorflow::PartialTensorShape({1, 2});
  ts3.dtype = tensorflow::DT_FLOAT;

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({ts1, ts2, ts3}));
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
