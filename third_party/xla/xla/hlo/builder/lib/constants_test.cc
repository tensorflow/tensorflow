/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/constants.h"

#include <limits>

#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ConstantsTest = ClientLibraryTestBase;

using ::testing::HasSubstr;

XLA_TEST_F(ConstantsTest, ConstantR0WithTypeS32) {
  XlaBuilder builder(TestName());
  ConstantR0WithType(&builder, xla::S32, 4);
  ComputeAndCompareR0<int32_t>(&builder, 4, {});
}

XLA_TEST_F(ConstantsTest, ConstantR0WithTypeS32DoesNotAcceptFloats) {
  XlaBuilder builder(TestName());
  ConstantR0WithType(&builder, xla::S32, 4.5);
  auto statusor = builder.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("Invalid cast"));
}

XLA_TEST_F(ConstantsTest, ConstantR0WithTypeF32) {
  XlaBuilder builder(TestName());
  ConstantR0WithType(&builder, xla::F32, -7);
  ComputeAndCompareR0<float>(&builder, -7, {});
  ConstantR0WithType(&builder, xla::F32, 0.5);
  ComputeAndCompareR0<float>(&builder, 0.5, {});
}

XLA_TEST_F(ConstantsTest, ScalarLikeS32) {
  XlaBuilder builder(TestName());
  ScalarLike(ConstantR0<int32_t>(&builder, 42), -3);
  ComputeAndCompareR0<int32_t>(&builder, -3, {});
}

XLA_TEST_F(ConstantsTest, ScalarLikeF32) {
  XlaBuilder builder(TestName());
  ScalarLike(ConstantR0<float>(&builder, 42.75), -3.2);
  ComputeAndCompareR0<float>(&builder, -3.2, {});
}

XLA_TEST_F(ConstantsTest, ZeroS32) {
  XlaBuilder builder(TestName());
  Zero(&builder, S32);
  ComputeAndCompareR0<int32_t>(&builder, 0, {});
}

XLA_TEST_F(ConstantsTest, ZeroF32) {
  XlaBuilder builder(TestName());
  Zero(&builder, F32);
  ComputeAndCompareR0<float>(&builder, 0.0, {});
}

XLA_TEST_F(ConstantsTest, ZerosS32) {
  XlaBuilder builder(TestName());
  Zeros(&builder, ShapeUtil::MakeShape(S32, {2, 2}));
  ComputeAndCompareR2<int32_t>(&builder, {{0, 0}, {0, 0}}, {});
}

XLA_TEST_F(ConstantsTest, ZerosLikeF32) {
  XlaBuilder builder(TestName());
  ZerosLike(ConstantR1<float>(&builder, {1., 2., 3.}));
  ComputeAndCompareR1<float>(&builder, {0., 0., 0.}, {});
}

XLA_TEST_F(ConstantsTest, OneS32) {
  XlaBuilder builder(TestName());
  One(&builder, S32);
  ComputeAndCompareR0<int32_t>(&builder, 1, {});
}

XLA_TEST_F(ConstantsTest, OneF32) {
  XlaBuilder builder(TestName());
  One(&builder, F32);
  ComputeAndCompareR0<float>(&builder, 1., {});
}

XLA_TEST_F(ConstantsTest, EpsilonF32) {
  XlaBuilder builder(TestName());
  Epsilon(&builder, F32);
  ComputeAndCompareR0<float>(&builder, std::numeric_limits<float>::epsilon(),
                             {});
}

XLA_TEST_F(ConstantsTest, MinFiniteValueS32) {
  XlaBuilder builder(TestName());
  MinFiniteValue(&builder, S32);
  ComputeAndCompareR0<int32_t>(&builder, std::numeric_limits<int32_t>::min(),
                               {});
}

XLA_TEST_F(ConstantsTest, MaxFiniteValueS32) {
  XlaBuilder builder(TestName());
  MaxFiniteValue(&builder, S32);
  ComputeAndCompareR0<int32_t>(&builder, std::numeric_limits<int32_t>::max(),
                               {});
}

XLA_TEST_F(ConstantsTest, MinFiniteValueF32) {
  XlaBuilder builder(TestName());
  MinFiniteValue(&builder, F32);
  ComputeAndCompareR0<float>(&builder, -std::numeric_limits<float>::max(), {});
}

XLA_TEST_F(ConstantsTest, MaxFiniteValueF32) {
  XlaBuilder builder(TestName());
  MaxFiniteValue(&builder, F32);
  ComputeAndCompareR0<float>(&builder, std::numeric_limits<float>::max(), {});
}

XLA_TEST_F(ConstantsTest, MinValueS32) {
  XlaBuilder builder(TestName());
  MinValue(&builder, S32);
  ComputeAndCompareR0<int32_t>(&builder, std::numeric_limits<int32_t>::min(),
                               {});
}

XLA_TEST_F(ConstantsTest, MaxValueS32) {
  XlaBuilder builder(TestName());
  MaxValue(&builder, S32);
  ComputeAndCompareR0<int32_t>(&builder, std::numeric_limits<int32_t>::max(),
                               {});
}

XLA_TEST_F(ConstantsTest, MinValueF32) {
  XlaBuilder builder(TestName());
  MinValue(&builder, F32);
  ComputeAndCompareR0<float>(&builder, -std::numeric_limits<float>::infinity(),
                             {});
}

XLA_TEST_F(ConstantsTest, MaxValueF32) {
  XlaBuilder builder(TestName());
  MaxValue(&builder, F32);
  ComputeAndCompareR0<float>(&builder, std::numeric_limits<float>::infinity(),
                             {});
}

XLA_TEST_F(ConstantsTest, NanValueF32) {
  XlaBuilder builder(TestName());
  NanValue(&builder, F32);
  ComputeAndCompareR0<float>(&builder, std::numeric_limits<float>::quiet_NaN(),
                             {});
}

}  // namespace
}  // namespace xla
