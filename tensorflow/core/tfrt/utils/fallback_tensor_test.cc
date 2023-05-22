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
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tensorflow {
namespace tfrt_stub {
namespace {

TEST(FallbackTensorTest, ImmutableTensor) {
  int32_t scalar = 123;
  tensorflow::Tensor tensor(scalar);

  auto immutable_tensor = ImmutableTensor::Create(tensor);

  ASSERT_EQ(immutable_tensor.tensor().NumElements(), 1);
  ASSERT_EQ(immutable_tensor.tensor().dtype(), tensorflow::DT_INT32);
  auto flat = immutable_tensor.tensor().flat<int32_t>();
  EXPECT_EQ(flat(0), 123);
  EXPECT_FALSE(immutable_tensor.tensor().RefCountIsOne());
  EXPECT_EQ(tensor.TotalBytes(), immutable_tensor.tensor().TotalBytes());
}

TEST(FallbackTensorTest, StringImmutableTensor) {
  tensorflow::tstring scalar = "string";
  tensorflow::Tensor tensor(scalar);

  auto immutable_tensor = ImmutableTensor::Create(tensor);

  ASSERT_EQ(immutable_tensor.tensor().NumElements(), 1);
  ASSERT_EQ(immutable_tensor.tensor().dtype(), tensorflow::DT_STRING);
  auto flat = immutable_tensor.tensor().flat<tensorflow::tstring>();
  EXPECT_EQ(flat(0), "string");
  EXPECT_FALSE(immutable_tensor.tensor().RefCountIsOne());
  EXPECT_EQ(tensor.TotalBytes(), immutable_tensor.tensor().TotalBytes());
}

TEST(FallbackTensorTest, FallbackTensor) {
  int32_t scalar = 123;
  tensorflow::Tensor tensor(scalar);

  {
    FallbackTensor fallback_tensor(tensor);
    EXPECT_FALSE(fallback_tensor.is_immutable());

    ASSERT_EQ(fallback_tensor.tensor().NumElements(), 1);
    ASSERT_EQ(fallback_tensor.tensor().dtype(), tensorflow::DT_INT32);
    auto flat = fallback_tensor.tensor().flat<int32_t>();
    EXPECT_EQ(flat(0), 123);

    FallbackTensor copy(fallback_tensor);
    FallbackTensor assign;
    assign = fallback_tensor;

    ASSERT_EQ(copy.tensor().NumElements(), 1);
    ASSERT_EQ(copy.tensor().dtype(), tensorflow::DT_INT32);
    EXPECT_EQ(copy.tensor().flat<int32_t>()(0), 123);
    ASSERT_EQ(assign.tensor().NumElements(), 1);
    ASSERT_EQ(assign.tensor().dtype(), tensorflow::DT_INT32);
    EXPECT_EQ(assign.tensor().flat<int32_t>()(0), 123);

    fallback_tensor = {};

    ASSERT_EQ(copy.tensor().NumElements(), 1);
    ASSERT_EQ(copy.tensor().dtype(), tensorflow::DT_INT32);
    EXPECT_EQ(copy.tensor().flat<int32_t>()(0), 123);
    ASSERT_EQ(assign.tensor().NumElements(), 1);
    ASSERT_EQ(assign.tensor().dtype(), tensorflow::DT_INT32);
    EXPECT_EQ(assign.tensor().flat<int32_t>()(0), 123);
  }

  auto immutable_tensor = ImmutableTensor::Create(tensor);

  {
    FallbackTensor fallback_tensor(&immutable_tensor);
    EXPECT_TRUE(fallback_tensor.is_immutable());

    ASSERT_EQ(fallback_tensor.tensor().NumElements(), 1);
    ASSERT_EQ(fallback_tensor.tensor().dtype(), tensorflow::DT_INT32);
    auto flat = fallback_tensor.tensor().flat<int32_t>();
    EXPECT_EQ(flat(0), 123);

    FallbackTensor copy(fallback_tensor);
    FallbackTensor assign;
    assign = fallback_tensor;

    ASSERT_EQ(copy.tensor().NumElements(), 1);
    ASSERT_EQ(copy.tensor().dtype(), tensorflow::DT_INT32);
    EXPECT_EQ(copy.tensor().flat<int32_t>()(0), 123);
    ASSERT_EQ(assign.tensor().NumElements(), 1);
    ASSERT_EQ(assign.tensor().dtype(), tensorflow::DT_INT32);
    EXPECT_EQ(assign.tensor().flat<int32_t>()(0), 123);

    fallback_tensor = {};

    ASSERT_EQ(copy.tensor().NumElements(), 1);
    ASSERT_EQ(copy.tensor().dtype(), tensorflow::DT_INT32);
    EXPECT_EQ(copy.tensor().flat<int32_t>()(0), 123);
    ASSERT_EQ(assign.tensor().NumElements(), 1);
    ASSERT_EQ(assign.tensor().dtype(), tensorflow::DT_INT32);
    EXPECT_EQ(assign.tensor().flat<int32_t>()(0), 123);
  }
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
