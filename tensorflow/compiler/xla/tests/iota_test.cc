/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace {

template <typename T>
std::vector<T> GetR1Expected(const int64 num_elements) {
  std::vector<T> result(num_elements);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

class IotaR1Test
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<std::tuple<PrimitiveType, int>> {};

TEST_P(IotaR1Test, DoIt) {
  const auto& spec = GetParam();
  const auto element_type = std::get<0>(spec);
  const int64 num_elements = std::get<1>(spec);
  XlaBuilder builder(TestName() + "_" + PrimitiveType_Name(element_type));
  Iota(&builder, element_type, num_elements);
  if (element_type == F32) {
    ComputeAndCompareR1<float>(&builder, GetR1Expected<float>(num_elements), {},
                               ErrorSpec{0.0001});
  } else if (element_type == U32) {
    ComputeAndCompareR1<uint32>(&builder, GetR1Expected<uint32>(num_elements),
                                {});
  } else {
    CHECK_EQ(element_type, S32);
    ComputeAndCompareR1<int32>(&builder, GetR1Expected<int32>(num_elements),
                               {});
  }
}

INSTANTIATE_TEST_CASE_P(IotaR1TestInstantiation, IotaR1Test,
                        ::testing::Combine(::testing::Values(F32, U32, S32),
                                           ::testing::Range(/*start=*/10,
                                                            /*end=*/10001,
                                                            /*step=*/10)));

class IotaR2Test : public ClientLibraryTestBase,
                   public ::testing::WithParamInterface<
                       std::tuple<PrimitiveType, int, int>> {};

TEST_P(IotaR2Test, DoIt) {
  const auto& spec = GetParam();
  const auto element_type = std::get<0>(spec);
  const int64 num_elements = std::get<1>(spec);
  const int64 iota_dim = std::get<2>(spec);
  XlaBuilder builder(TestName() + "_" + PrimitiveType_Name(element_type));
  std::vector<int64> dimensions = {42};
  dimensions.insert(dimensions.begin() + iota_dim, num_elements);
  Iota(&builder, ShapeUtil::MakeShape(element_type, dimensions), iota_dim);
  if (primitive_util::IsFloatingPointType(element_type)) {
    ComputeAndCompare(&builder, {}, ErrorSpec{0.0001});
  } else {
    ComputeAndCompare(&builder, {});
  }
}

INSTANTIATE_TEST_CASE_P(IotaR2TestInstantiation, IotaR2Test,
                        ::testing::Combine(::testing::Values(F32, S32),
                                           ::testing::Range(/*start=*/10,
                                                            /*end=*/1001,
                                                            /*step=*/10),
                                           ::testing::Values(0, 1)));

class IotaR3Test : public ClientLibraryTestBase,
                   public ::testing::WithParamInterface<
                       std::tuple<PrimitiveType, int, int>> {};

TEST_P(IotaR3Test, DoIt) {
  const auto& spec = GetParam();
  const auto element_type = std::get<0>(spec);
  const int64 num_elements = std::get<1>(spec);
  const int64 iota_dim = std::get<2>(spec);
  XlaBuilder builder(TestName() + "_" + PrimitiveType_Name(element_type));
  std::vector<int64> dimensions = {42, 19};
  dimensions.insert(dimensions.begin() + iota_dim, num_elements);
  Iota(&builder, ShapeUtil::MakeShape(element_type, dimensions), iota_dim);
  if (primitive_util::IsFloatingPointType(element_type)) {
    ComputeAndCompare(&builder, {}, ErrorSpec{0.0001});
  } else {
    ComputeAndCompare(&builder, {});
  }
}

INSTANTIATE_TEST_CASE_P(IotaR3TestInstantiation, IotaR3Test,
                        ::testing::Combine(::testing::Values(F32, S32),
                                           ::testing::Range(/*start=*/10,
                                                            /*end=*/1001,
                                                            /*step=*/10),
                                           ::testing::Values(0, 1, 2)));

class IotaR3PredTest : public ClientLibraryTestBase,
                       public ::testing::WithParamInterface<int> {};

TEST_P(IotaR3PredTest, DoIt) {
  const auto element_type = PRED;
  const int64 num_elements = 2;
  const int64 iota_dim = GetParam();
  XlaBuilder builder(TestName() + "_" + PrimitiveType_Name(element_type));
  std::vector<int64> dimensions = {42, 19};
  dimensions.insert(dimensions.begin() + iota_dim, num_elements);
  Iota(&builder, ShapeUtil::MakeShape(element_type, dimensions), iota_dim);
  if (primitive_util::IsFloatingPointType(element_type)) {
    ComputeAndCompare(&builder, {}, ErrorSpec{0.0001});
  } else {
    ComputeAndCompare(&builder, {});
  }
}

INSTANTIATE_TEST_CASE_P(IotaR3PredTestInstantiation, IotaR3PredTest,
                        ::testing::Values(0, 1, 2));

}  // namespace
}  // namespace xla
