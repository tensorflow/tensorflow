/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/literal_util.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(LiteralUtil, LiteralToHostTensor) {
  // int64 literal can only be converted to an int64 host tensor.
  std::vector<int64_t> int64_values = {1, 2, 3};
  xla::Literal int64_values_literal =
      xla::LiteralUtil::CreateR1(absl::Span<const int64_t>(int64_values));
  Tensor host_tensor;
  EXPECT_EQ("Cannot convert literal of type S64 to tensor of type int32",
            LiteralToHostTensor(int64_values_literal, DT_INT32, &host_tensor)
                .message());
  EXPECT_EQ("Cannot convert literal of type S64 to tensor of type qint32",
            LiteralToHostTensor(int64_values_literal, DT_QINT32, &host_tensor)
                .message());
  EXPECT_TRUE(
      LiteralToHostTensor(int64_values_literal, DT_INT64, &host_tensor).ok());
  test::ExpectTensorEqual<int64_t>(host_tensor,
                                   test::AsTensor<int64_t>(int64_values));
}

template <class T>
using LiteralUtilTest = ::testing::Test;
using Types =
    ::testing::Types<std::pair<int8, qint8>, std::pair<uint8, quint8>,
                     std::pair<int16, qint16>, std::pair<uint16, quint16>,
                     std::pair<int32, qint32>>;

TYPED_TEST_SUITE(LiteralUtilTest, Types);

TYPED_TEST(LiteralUtilTest, LiteralToQuantizedHostTensor) {
  using int_type = typename TypeParam::first_type;
  using qint_type = typename TypeParam::second_type;

  Tensor host_tensor;
  std::vector<int_type> int_values = {10, 11};
  xla::Literal int_values_literal =
      xla::LiteralUtil::CreateR1(absl::Span<const int_type>(int_values));
  EXPECT_TRUE(LiteralToHostTensor(int_values_literal,
                                  DataTypeToEnum<int_type>::value, &host_tensor)
                  .ok());
  test::ExpectTensorEqual<int_type>(host_tensor,
                                    test::AsTensor<int_type>(int_values));

  EXPECT_TRUE(LiteralToHostTensor(int_values_literal,
                                  DataTypeToEnum<qint_type>::value,
                                  &host_tensor)
                  .ok());
  std::vector<qint_type> qint_values = {10, 11};
  test::ExpectTensorEqual<qint_type>(host_tensor,
                                     test::AsTensor<qint_type>(qint_values));

  EXPECT_EQ(
      error::INVALID_ARGUMENT,
      LiteralToHostTensor(int_values_literal, DT_INT64, &host_tensor).code());
}

}  // namespace
}  // namespace tensorflow
