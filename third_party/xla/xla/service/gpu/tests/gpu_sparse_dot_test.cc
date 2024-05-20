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

#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "third_party/half/half.hpp"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"

namespace xla {
namespace gpu {
namespace {

class SparseDotTest
    : public GpuCodegenTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int>> {
 protected:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(true);
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }

  // Combinations of 2-item indices in the 4-group, where the first index is
  // less than the second.
  const int indices_2_in_4_[6] = {0b0100, 0b1000, 0b1100,
                                  0b1001, 0b1101, 0b1110};

  std::vector<uint16_t> CreateInput(int width, int height) {
    std::vector<uint16_t> input(width * height);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int value = (i * 2 + j) % 100 + 1;
        input[i * width + j] =
            half_float::detail::int2half<std::round_to_nearest>(value);
      }
    }
    return input;
  }

  std::vector<uint16_t> CreateMeta(int width, int height) {
    std::vector<uint16_t> meta(width * height);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        uint16_t bitmask = 0;
        for (int k = 0; k < 4; ++k) {
          int index = (i + j * 2 + k * 3) % 6;
          bitmask |= indices_2_in_4_[index] << (k * 4);
        }
        meta[i * width + j] = bitmask;
      }
    }
    return meta;
  }

  std::vector<uint16_t> Sparsify(absl::Span<uint16_t> input,
                                 absl::Span<uint16_t> meta) {
    std::vector<uint16_t> result(input.size());
    for (int i = 0; i < input.size(); i += 16) {
      for (int j = 0; j < 4; ++j) {
        int mask = meta[i / 16] >> (j * 4);
        int p1 = i + j * 4 + (mask & 0b0011);
        int p2 = i + j * 4 + (mask & 0b1100) / 4;
        result[p1] = input[p1];
        result[p2] = input[p2];
      }
    }
    return result;
  }

  std::vector<uint16_t> Compress(absl::Span<uint16_t> input) {
    std::vector<uint16_t> result;
    for (uint16_t value : input) {
      if (value != 0) result.push_back(value);
    }
    return result;
  }
};

TEST_P(SparseDotTest, CompareWithDense) {
  int m, n, k;
  std::tie(m, n, k) = GetParam();

  auto in1 = CreateInput(m, k);
  auto in2 = CreateInput(n, k);
  auto meta = CreateMeta(m, k / 16);
  auto sparse_zeros = Sparsify(absl::MakeSpan(in1), absl::MakeSpan(meta));
  auto sparse_packed = Compress(absl::MakeSpan(sparse_zeros));

  // Execute dense dot.
  const char* kDenseTpl = R"(
HloModule TestDense

ENTRY main {
  lhs = f16[$0,$1] parameter(0)
  rhs = f16[$2,$1] parameter(1)
  ROOT dot = f32[$0,$2] dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";
  auto dense_hlo = absl::Substitute(kDenseTpl, m, k, n);

  Literal dense_lhs =
      LiteralUtil::CreateR1<uint16_t>(absl::MakeSpan(sparse_zeros));
  Literal dense_rhs = LiteralUtil::CreateR1<uint16_t>(absl::MakeSpan(in2));

  auto dense_module = ParseAndReturnVerifiedModule(dense_hlo);
  EXPECT_OK(dense_module);
  auto dense_result =
      Execute(std::move(*dense_module), {&dense_lhs, &dense_rhs});
  EXPECT_OK(dense_result);

  // Execute sparse dot.
  const char* kSparseTpl = R"(
HloModule TestSparse

ENTRY main {
  lhs = f16[$0,$1] parameter(0)
  rhs = f16[$2,$3] parameter(1)
  meta = u16[$0,$4] parameter(2)
  ROOT dot = f32[$0,$2] dot(lhs, rhs, meta),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}, sparsity=L.1@2:4
})";
  auto sparse_hlo = absl::Substitute(kSparseTpl, m, k / 2, n, k, k / 16);

  Literal sparse_lhs =
      LiteralUtil::CreateR1<uint16_t>(absl::MakeSpan(sparse_packed));
  Literal sparse_rhs = LiteralUtil::CreateR1<uint16_t>(absl::MakeSpan(in2));
  Literal sparse_meta = LiteralUtil::CreateR1<uint16_t>(absl::MakeSpan(meta));

  auto sparse_module = ParseAndReturnVerifiedModule(sparse_hlo);
  EXPECT_OK(sparse_module);
  auto sparse_result = Execute(std::move(*sparse_module),
                               {&sparse_lhs, &sparse_rhs, &sparse_meta});
  EXPECT_OK(sparse_result);

  // Compare the results.
  EXPECT_EQ(*dense_result, *sparse_result);
}

INSTANTIATE_TEST_SUITE_P(
    Sparsity, SparseDotTest,
    ::testing::Combine(/*m=*/::testing::Values(32, 256),
                       /*n=*/::testing::Values(32, 256),
                       /*k=*/::testing::Values(64, 512)),
    [](const ::testing::TestParamInfo<SparseDotTest::ParamType>& info) {
      return absl::StrCat("m", std::get<0>(info.param), "n",
                          std::get<1>(info.param), "k",
                          std::get<2>(info.param));
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
