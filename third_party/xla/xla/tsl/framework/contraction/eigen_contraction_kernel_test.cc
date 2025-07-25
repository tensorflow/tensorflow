/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tsl/framework/contraction/eigen_contraction_kernel.h"  // IWYU pragma: keep

#include <array>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace {

template <typename T>
static void ExpectClose(Eigen::Tensor<T, 2> res, Eigen::Tensor<T, 2> expected) {
  T atol = T(5.0) * Eigen::NumTraits<T>::epsilon();  // Absolute tolerance.
  EXPECT_EQ(res.dimensions(), expected.dimensions());
  for (int i = 0; i < res.dimension(0); ++i) {
    for (int j = 0; j < res.dimension(1); ++j) {
      EXPECT_NEAR(res(i, j), expected(i, j), atol);
    }
  }
}

template <typename LhsType, typename RhsType, typename OutType>
void RunEigenMatMul(int m, int k, int n) {
  Eigen::Tensor<LhsType, 2> lhs(m, k);
  Eigen::Tensor<RhsType, 2> rhs(k, n);
  Eigen::Tensor<OutType, 2> out(m, n);

  lhs.setRandom();
  rhs.setRandom();
  out.setZero();

  using DimPair = typename Eigen::Tensor<LhsType, 2>::DimensionPair;
  std::array<DimPair, 1> dims({DimPair(1, 0)});
  out = lhs.contract(rhs, dims);

  Eigen::Tensor<OutType, 2> expected =
      lhs.template cast<OutType>().contract(rhs.template cast<OutType>(), dims);

  ExpectClose(out, expected);
}

struct EigenContractionKernelTestParams {
  int m;
  int k;
  int n;
};

class EigenContractionKernelTest
    : public testing::TestWithParam<EigenContractionKernelTestParams> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<EigenContractionKernelTestParams>& info) {
    return absl::StrCat(info.param.m, "x", info.param.k, "x", info.param.n);
  }
};

TEST_P(EigenContractionKernelTest, S8S8S32) {
  EigenContractionKernelTestParams param = GetParam();
  RunEigenMatMul<int8_t, int8_t, int32_t>(param.m, param.k, param.n);
}

INSTANTIATE_TEST_SUITE_P(
    EigenContractionKernelTestSuite, EigenContractionKernelTest,
    testing::ValuesIn<EigenContractionKernelTestParams>({{10, 10, 10},
                                                         {128, 128, 128},
                                                         {64, 1024, 64},
                                                         {1, 64, 64},
                                                         {256, 1, 128},
                                                         {512, 128, 1}}),
    EigenContractionKernelTest::Name);

}  // namespace
