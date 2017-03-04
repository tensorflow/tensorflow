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

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(LiteralUtil, LiteralToHostTensor) {
  // int64 literal can only be converted to an int64 host tensor.
  {
    std::vector<int64> int64_values = {1, 2, 3};
    std::unique_ptr<xla::Literal> int64_values_literal =
        xla::LiteralUtil::CreateR1(gtl::ArraySlice<int64>(int64_values));
    Tensor host_tensor;
    EXPECT_EQ("Cannot convert literal of type S64 to tensor of type int32",
              LiteralToHostTensor(*int64_values_literal, DT_INT32, &host_tensor)
                  .error_message());
    EXPECT_EQ(
        "Cannot convert literal of type S64 to tensor of type qint32",
        LiteralToHostTensor(*int64_values_literal, DT_QINT32, &host_tensor)
            .error_message());
    EXPECT_TRUE(
        LiteralToHostTensor(*int64_values_literal, DT_INT64, &host_tensor)
            .ok());
    test::ExpectTensorEqual<int64>(host_tensor,
                                   test::AsTensor<int64>(int64_values));
  }

  {
    // Repeat tests with int32.
    Tensor host_tensor;
    std::vector<int32> int32_values = {10, 11};
    std::unique_ptr<xla::Literal> int32_values_literal =
        xla::LiteralUtil::CreateR1(gtl::ArraySlice<int32>(int32_values));
    EXPECT_TRUE(
        LiteralToHostTensor(*int32_values_literal, DT_INT32, &host_tensor)
            .ok());
    test::ExpectTensorEqual<int32>(host_tensor,
                                   test::AsTensor<int32>(int32_values));

    EXPECT_TRUE(
        LiteralToHostTensor(*int32_values_literal, DT_QINT32, &host_tensor)
            .ok());
    std::vector<qint32> qint32_values = {10, 11};
    test::ExpectTensorEqual<qint32>(host_tensor,
                                    test::AsTensor<qint32>(qint32_values));

    EXPECT_EQ("Cannot convert literal of type S32 to tensor of type int64",
              LiteralToHostTensor(*int32_values_literal, DT_INT64, &host_tensor)
                  .error_message());
  }
}

}  // namespace tensorflow
