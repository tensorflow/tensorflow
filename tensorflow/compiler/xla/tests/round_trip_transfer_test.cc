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

// Tests transferring literals of various shapes and values in and out of the
// XLA service.

#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class RoundTripTransferTest : public ClientLibraryTestBase {
 protected:
  void RoundTripTest(const Literal& original) {
    std::unique_ptr<GlobalData> data =
        client_->TransferToServer(original).ConsumeValueOrDie();
    Literal result = client_->Transfer(*data).ConsumeValueOrDie();
    EXPECT_TRUE(LiteralTestUtil::Equal(original, result));
  }
};

TEST_F(RoundTripTransferTest, R0S32) {
  RoundTripTest(LiteralUtil::CreateR0<int32_t>(42));
}

TEST_F(RoundTripTransferTest, R0F32) {
  RoundTripTest(LiteralUtil::CreateR0<float>(42.0));
}

TEST_F(RoundTripTransferTest, R1F32_Len0) {
  RoundTripTest(LiteralUtil::CreateR1<float>({}));
}

TEST_F(RoundTripTransferTest, R1F32_Len2) {
  RoundTripTest(LiteralUtil::CreateR1<float>({42.0, 64.0}));
}

TEST_F(RoundTripTransferTest, R1F32_Len256) {
  std::vector<float> values(256);
  std::iota(values.begin(), values.end(), 1.0);
  RoundTripTest(LiteralUtil::CreateR1<float>(values));
}

TEST_F(RoundTripTransferTest, R1F32_Len1024) {
  std::vector<float> values(1024);
  std::iota(values.begin(), values.end(), 1.0);
  RoundTripTest(LiteralUtil::CreateR1<float>(values));
}

TEST_F(RoundTripTransferTest, R1F32_Len1025) {
  std::vector<float> values(1025);
  std::iota(values.begin(), values.end(), 1.0);
  RoundTripTest(LiteralUtil::CreateR1<float>(values));
}

TEST_F(RoundTripTransferTest, R1F32_Len4096) {
  std::vector<float> values(4096);
  std::iota(values.begin(), values.end(), 1.0);
  RoundTripTest(LiteralUtil::CreateR1<float>(values));
}

TEST_F(RoundTripTransferTest, R2F32_Len10x0) {
  RoundTripTest(LiteralUtil::CreateR2FromArray2D<float>(Array2D<float>(10, 0)));
}

TEST_F(RoundTripTransferTest, R2F32_Len2x2) {
  RoundTripTest(LiteralUtil::CreateR2<float>({{42.0, 64.0}, {77.0, 88.0}}));
}

TEST_F(RoundTripTransferTest, R3F32) {
  RoundTripTest(
      LiteralUtil::CreateR3<float>({{{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}},
                                    {{3.0, 4.0}, {3.0, 4.0}, {3.0, 4.0}}}));
}

TEST_F(RoundTripTransferTest, R4F32) {
  RoundTripTest(LiteralUtil::CreateR4<float>({{
      {{10, 11, 12, 13}, {14, 15, 16, 17}},
      {{18, 19, 20, 21}, {22, 23, 24, 25}},
      {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }}));
}

TEST_F(RoundTripTransferTest, EmptyTuple) {
  RoundTripTest(LiteralUtil::MakeTuple({}));
}

TEST_F(RoundTripTransferTest, TupleOfR1F32) {
  RoundTripTest(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({1, 2}),
                                        LiteralUtil::CreateR1<float>({3, 4})}));
}

TEST_F(RoundTripTransferTest, TupleOfR1F32_Len0_Len2) {
  RoundTripTest(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({}),
                                        LiteralUtil::CreateR1<float>({3, 4})}));
}

TEST_F(RoundTripTransferTest, TupleOfR0F32AndR1S32) {
  RoundTripTest(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(1.0), LiteralUtil::CreateR1<int>({2, 3})}));
}

// Below two tests are added to identify the cost of large data transfers.
TEST_F(RoundTripTransferTest, R2F32_Large) {
  RoundTripTest(LiteralUtil::CreateR2F32Linspace(-1.0f, 1.0f, 512, 512));
}

TEST_F(RoundTripTransferTest, R4F32_Large) {
  Array4D<float> array4d(2, 2, 256, 256);
  array4d.FillWithMultiples(1.0f);
  RoundTripTest(LiteralUtil::CreateR4FromArray4D<float>(array4d));
}

}  // namespace
}  // namespace xla
