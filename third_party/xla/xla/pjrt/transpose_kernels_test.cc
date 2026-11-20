/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/transpose_kernels.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/strings/str_format.h"
#include "hwy/targets.h"
#include "xla/hlo/testlib/test.h"
#include "xla/types.h"

namespace xla {
namespace {

template <typename T, int bs>
constexpr int OuterBlockSize() {
  return std::max<int>(1, kMaxOuterBlockSizeBytes / (bs * sizeof(T)));
}

// Returns the byte offset of the starting element within a contiguous buffer of
// `extent` items spaced by `stride` bytes. For negative strides, starts at the
// last slot so stepping backward stays within `[0, extent * |stride|)`.
int64_t StridedStart(int64_t extent, int64_t stride) {
  return (stride < 0 && extent > 0) ? (extent - 1) * (-stride) : 0;
}

// Checks that `fn` transposes `count` slices of `rows x cols` elements of size
// `elem_size` with byte row strides `lda` and `ldb` (which may be negative).
void CheckTransposeKernel(TransposeOp::Fn fn, size_t elem_size, int64_t rows,
                          int64_t cols, int64_t lda, int64_t ldb,
                          int64_t count = 1, size_t byte_offset = 0) {
  const int64_t slice_a = rows * std::abs(lda);
  const int64_t slice_b = cols * std::abs(ldb);

  std::vector<uint8_t> input(byte_offset +
                             static_cast<size_t>(count * slice_a));
  for (size_t i = byte_offset; i < input.size(); ++i) {
    input[i] = static_cast<uint8_t>((i * 251 + 17 + (i >> 7)) & 0xff);
  }
  std::vector<uint8_t> expected(
      byte_offset + static_cast<size_t>(count * slice_b), 0x5a);
  std::vector<uint8_t> actual(expected.size(), 0x5a);

  const bool backward = lda < 0 || ldb < 0;
  const int64_t da = count > 1 ? (backward ? -slice_a : slice_a) : 0;
  const int64_t db = count > 1 ? (backward ? -slice_b : slice_b) : 0;

  const char* a = reinterpret_cast<const char*>(input.data() + byte_offset);
  char* b_expected = reinterpret_cast<char*>(expected.data() + byte_offset);
  char* b_actual = reinterpret_cast<char*>(actual.data() + byte_offset);
  if (count > 0) {
    a += StridedStart(count, da) + StridedStart(rows, lda);
    const int64_t b_start = StridedStart(count, db) + StridedStart(cols, ldb);
    b_expected += b_start;
    b_actual += b_start;
  }

  for (int64_t s = 0; s < count; ++s) {
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        std::memcpy(b_expected + s * db + c * ldb + r * elem_size,
                    a + s * da + r * lda + c * elem_size, elem_size);
      }
    }
  }

  TransposeOp op;
  op.count = count;
  op.da = da;
  op.db = db;
  op.lda = lda;
  op.ldb = ldb;
  op.a_elems = static_cast<uint16_t>(cols);
  op.b_elems = static_cast<uint16_t>(rows);
  fn(&op, a, b_actual, /*scratch=*/nullptr);

  EXPECT_EQ(actual, expected)
      << "elem_size=" << elem_size << " rows=" << rows << " cols=" << cols
      << " lda=" << lda << " ldb=" << ldb << " count=" << count
      << " byte_offset=" << byte_offset;
}

template <typename T, int bs>
void CheckSquareKernel(int outer_bs) {
  constexpr int packed = sizeof(T) | (bs << 16);
  const int64_t n = outer_bs * bs;
  const int64_t row_bytes = n * static_cast<int64_t>(sizeof(T));
  // Test both tightly-packed rows (0 padding) and rows with 64 bytes of
  // padding (which exercises wide-stride / rectangular kernels).
  for (int64_t lda : {row_bytes, row_bytes + 64}) {
    for (int64_t ldb : {row_bytes, row_bytes + 64}) {
      CheckTransposeKernel(GetTransposeMacroKernelOp<packed>(), sizeof(T), n, n,
                           lda, ldb);
    }
  }
}

template <typename T, int bs>
void TestMicroKernelEquivalence() {
  CheckSquareKernel<T, bs>(/*outer_bs=*/1);
}

template <typename T, int bs>
void TestMacroKernelEquivalence() {
  CheckSquareKernel<T, bs>(OuterBlockSize<T, bs>());
}

TEST(TransposeMicroKernelTest, ExactEquivalence) {
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
    hwy::SetSupportedTargetsForTest(target);

    // TransposeMicroKernel256Square is triggered when a logical row of the tile
    // (bs * sizeof(T)) is 256 bits (32 bytes).
    TestMicroKernelEquivalence<int8_t, 32>();
    TestMicroKernelEquivalence<int16_t, 16>();
    TestMicroKernelEquivalence<float, 8>();
    TestMicroKernelEquivalence<int64_t, 4>();
    TestMicroKernelEquivalence<absl::uint128, 2>();

    // TransposeMicroKernel128 or TransposeMicroKernel256Rect is triggered when
    // a logical row of the tile (bs * sizeof(T)) is 128 bits (16 bytes).
    // TransposeMicroKernel128 operates on 128-bit blocks when gathering from
    // memory (lda >= ldb && lda <= 64B) while TransposeMicroKernel256Rect packs
    // two rows into 256-bit vectors when scattering to memory (ldb > lda) or
    // for large strides.
    TestMicroKernelEquivalence<int8_t, 16>();
    TestMicroKernelEquivalence<int16_t, 8>();
    TestMicroKernelEquivalence<bfloat16, 8>();
    TestMicroKernelEquivalence<float, 4>();
    TestMicroKernelEquivalence<int64_t, 2>();

    // Smaller or larger cases fall back to TransposeMicroKernel128 or scalar
    // loop.
    TestMicroKernelEquivalence<int8_t, 8>();
    TestMicroKernelEquivalence<int16_t, 4>();
    TestMicroKernelEquivalence<bfloat16, 4>();
    TestMicroKernelEquivalence<float, 2>();
    TestMicroKernelEquivalence<int8_t, 4>();
    TestMicroKernelEquivalence<int16_t, 2>();
    TestMicroKernelEquivalence<int8_t, 2>();
    TestMicroKernelEquivalence<int8_t, 64>();
  }
  hwy::SetSupportedTargetsForTest(0);
}

// Same tile shapes as above, but transposing a macroblock of
// OuterBlockSize() x OuterBlockSize() tiles.
TEST(TransposeMacroKernelTest, ExactEquivalence) {
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
    hwy::SetSupportedTargetsForTest(target);

    TestMacroKernelEquivalence<int8_t, 32>();
    TestMacroKernelEquivalence<int16_t, 16>();
    TestMacroKernelEquivalence<float, 8>();
    TestMacroKernelEquivalence<int64_t, 4>();
    TestMacroKernelEquivalence<absl::uint128, 2>();

    TestMacroKernelEquivalence<int8_t, 16>();
    TestMacroKernelEquivalence<int16_t, 8>();
    TestMacroKernelEquivalence<bfloat16, 8>();
    TestMacroKernelEquivalence<float, 4>();
    TestMacroKernelEquivalence<int64_t, 2>();

    TestMacroKernelEquivalence<int8_t, 8>();
    TestMacroKernelEquivalence<int16_t, 4>();
    TestMacroKernelEquivalence<bfloat16, 4>();
    TestMacroKernelEquivalence<float, 2>();
    TestMacroKernelEquivalence<int8_t, 4>();
    TestMacroKernelEquivalence<int16_t, 2>();
    TestMacroKernelEquivalence<int8_t, 2>();
    TestMacroKernelEquivalence<int8_t, 64>();
  }
  hwy::SetSupportedTargetsForTest(0);
}

TEST(TransposeMacroKernelTest, InitializesHighwayTargetEagerly) {
  // Guard against returning Highway's lazy-init ChooseAndCall trampoline when
  // `HWY_DYNAMIC_POINTER_T` is queried before any Highway dispatch has run (or
  // after `DeInit()`). Because `TransposePlan` caches the returned function
  // pointer in `chunk_ops_`, storing the trampoline causes every subsequent
  // execution to re-run CPUID target detection.
  auto expect_eager_init = [](auto getter) {
    hwy::GetChosenTarget().DeInit();
    ASSERT_FALSE(hwy::GetChosenTarget().IsInitialized());
    const auto fn = getter();
    EXPECT_TRUE(hwy::GetChosenTarget().IsInitialized());
    EXPECT_EQ(fn, getter());
  };
  expect_eager_init(
      [] { return GetTransposeMacroKernelOp<sizeof(uint16_t) | (8 << 16)>(); });
  expect_eager_init([] { return GetPackSubbyteOp<4>(); });
  expect_eager_init([] { return GetDeinterleaveOp(sizeof(uint32_t), 2); });
}

struct DeinterleaveParam {
  // Highway target bit (`0` selects the plain scalar fallback for k >= 5).
  int64_t target;
  // Element size in bytes (1, 2, 4, or 8).
  size_t elem_size;
  // Number of interleaved output rows (2..7).
  int k;
  // Number of outer slices to process in one op call (`0`, `1`, or `3`).
  int64_t count;
  // Byte offset applied to input and output buffers to test pointer alignment.
  size_t byte_offset;
  // If true, output rows and slices stride backward (negative `ldb`/`da`/`db`).
  bool negative_ldb;
};

std::string FormatParam(const DeinterleaveParam& p) {
  std::string target =
      p.target == 0 ? "Fallback" : std::string(hwy::TargetName(p.target));
  for (char& c : target) {
    if (!std::isalnum(static_cast<unsigned char>(c))) {
      c = '_';
    }
  }
  return absl::StrFormat("%s_b%zu_k%d_c%d_off%zu_%s", target, p.elem_size, p.k,
                         p.count, p.byte_offset,
                         p.negative_ldb ? "neg" : "pos");
}

void PrintTo(const DeinterleaveParam& p, std::ostream* os) {
  *os << FormatParam(p);
}

std::string ParamName(const ::testing::TestParamInfo<DeinterleaveParam>& info) {
  return FormatParam(info.param);
}

std::vector<DeinterleaveParam> BuildDeinterleaveParams() {
  std::vector<DeinterleaveParam> params;
  auto add_variants = [&](int64_t target, size_t elem_size, int k) {
    for (int64_t count : {0, 1, 3}) {
      for (bool negative_ldb : {false, true}) {
        params.push_back(
            {target, elem_size, k, count, /*byte_offset=*/0, negative_ldb});
        params.push_back(
            {target, elem_size, k, count, /*byte_offset=*/1, negative_ldb});
        if (elem_size > 1) {
          params.push_back(
              {target, elem_size, k, count, elem_size, negative_ldb});
        }
      }
    }
  };
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
    for (size_t elem_size : {1, 2, 4, 8}) {
      for (int k : {2, 3, 4}) {
        add_variants(target, elem_size, k);
      }
    }
  }
  for (size_t elem_size : {1, 2, 4}) {
    for (int k : {5, 6, 7}) {
      if (DeinterleaveKernelIsFaster(elem_size, k)) {
        add_variants(/*target=*/0, elem_size, k);
      }
    }
  }
  return params;
}

class DeinterleaveKernelTest
    : public ::testing::TestWithParam<DeinterleaveParam> {
 protected:
  void TearDown() override { hwy::SetSupportedTargetsForTest(0); }
};

TEST_P(DeinterleaveKernelTest, MatchesReference) {
  const DeinterleaveParam& p = GetParam();
  if (p.target != 0) {
    hwy::SetSupportedTargetsForTest(p.target);
  }
  TransposeOp::Fn fn = GetDeinterleaveOp(p.elem_size, p.k);
  ASSERT_NE(fn, nullptr);

  const int64_t elem_size = static_cast<int64_t>(p.elem_size);
  const int64_t lda = p.k * elem_size;
  for (int64_t n :
       {0, 1, 2, 3, 4, 7, 8, 12, 15, 16, 17, 24, 31, 32, 33, 63, 64, 135}) {
    const int64_t row_bytes = n * elem_size;
    // Test tightly-packed rows, element-aligned padded rows (leaving 0x5a guard
    // bytes between rows), and unaligned row strides (for elem_size > 1,
    // exercising the stride alignment guard with aligned a/b pointers).
    const int64_t sign = p.negative_ldb ? -1 : 1;
    CheckTransposeKernel(fn, p.elem_size, n, p.k, lda, sign * row_bytes,
                         p.count, p.byte_offset);
    CheckTransposeKernel(fn, p.elem_size, n, p.k, lda, sign * (row_bytes + 16),
                         p.count, p.byte_offset);
    if (elem_size > 1) {
      CheckTransposeKernel(fn, p.elem_size, n, p.k, lda, sign * (row_bytes + 1),
                           p.count, p.byte_offset);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Deinterleave, DeinterleaveKernelTest,
                         ::testing::ValuesIn(BuildDeinterleaveParams()),
                         ParamName);

TEST(TransposeKernelsTest, DeinterleaveUnsupportedReturnsNull) {
  EXPECT_EQ(GetDeinterleaveOp(1, 1), nullptr);
  EXPECT_EQ(GetDeinterleaveOp(1, 8), nullptr);
  EXPECT_EQ(GetDeinterleaveOp(4, 6), nullptr);
  EXPECT_EQ(GetDeinterleaveOp(8, 5), nullptr);
  EXPECT_EQ(GetDeinterleaveOp(16, 2), nullptr);
}

}  // namespace
}  // namespace xla
