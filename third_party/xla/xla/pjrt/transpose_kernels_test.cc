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

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "hwy/targets.h"
#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

int OuterBlockSize(size_t elem_size, int bs) {
  return std::max<int>(
      1, kMaxOuterBlockSizeBytes / (bs * static_cast<int>(elem_size)));
}

// Returns the byte offset of the starting element within a contiguous buffer of
// `extent` items spaced by `stride` bytes. For negative strides, starts at the
// last slot so stepping backward stays within `[0, extent * |stride|)`.
int64_t StridedStart(int64_t extent, int64_t stride) {
  return (stride < 0 && extent > 0) ? (extent - 1) * (-stride) : 0;
}

// Checks that `fn` transposes `count` slices of `rows x cols` elements of size
// `elem_size` with byte row strides `lda` and `ldb` (`lda` may be negative or
// zero; `ldb` may be negative).
void CheckTransposeKernel(TransposeOp::Fn fn, size_t elem_size, int64_t rows,
                          int64_t cols, int64_t lda, int64_t ldb,
                          int64_t count = 1, size_t byte_offset = 0) {
  const int64_t slice_a =
      lda == 0 ? cols * static_cast<int64_t>(elem_size) : rows * std::abs(lda);
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

struct KernelShape {
  size_t elem_size;
  int bs;
};

// 512 bit (64 byte), 256 bit (32 byte), 128 bit (16 byte), and sub block rows.
constexpr KernelShape kKernelShapes[] = {
    {1, 64}, {2, 32}, {4, 16}, {8, 8}, {16, 4},  // 64 byte rows
    {1, 32}, {2, 16}, {4, 8},  {8, 4}, {16, 2},  // 32 byte rows
    {1, 16}, {2, 8},  {4, 4},  {8, 2},           // 16 byte rows
    {1, 8},  {2, 4},  {4, 2},                    // 8 byte rows
    {1, 4},  {2, 2},                             // 4 byte rows
    {1, 2},                                      // 2 byte rows
};

void CheckSquareKernel(size_t elem_size, int bs, int outer_bs) {
  const TransposeOp::Fn fn =
      GetTransposeMacroKernelOp(static_cast<int>(elem_size), bs);
  ASSERT_NE(fn, nullptr);
  const int64_t n = outer_bs * bs;
  const int64_t row_bytes = n * static_cast<int64_t>(elem_size);
  // Test both tightly-packed rows (0 padding) and rows with 64 bytes of
  // padding (which exercises wide-stride / rectangular kernels).
  for (int64_t lda : {row_bytes, row_bytes + 64}) {
    for (int64_t ldb : {row_bytes, row_bytes + 64}) {
      CheckTransposeKernel(fn, elem_size, n, n, lda, ldb);
    }
  }
}

TEST(TransposeMicroKernelTest, ExactEquivalence) {
  const absl::Cleanup reset_targets = [] {
    hwy::SetSupportedTargetsForTest(0);
  };
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
    SCOPED_TRACE(hwy::TargetName(target));
    hwy::SetSupportedTargetsForTest(target);
    for (const KernelShape& shape : kKernelShapes) {
      CheckSquareKernel(shape.elem_size, shape.bs, /*outer_bs=*/1);
    }
  }
}

TEST(TransposeMicroKernelTest, StrideDispatchBoundaries) {
  const absl::Cleanup reset_targets = [] {
    hwy::SetSupportedTargetsForTest(0);
  };
  auto check_byte_strides = [](size_t elem_size, int bs,
                               absl::Span<const int64_t> in_byte_strides,
                               absl::Span<const int64_t> out_byte_strides) {
    const TransposeOp::Fn fn =
        GetTransposeMacroKernelOp(static_cast<int>(elem_size), bs);
    ASSERT_NE(fn, nullptr);
    for (int64_t lda : in_byte_strides) {
      for (int64_t ldb : out_byte_strides) {
        CheckTransposeKernel(fn, elem_size, bs, bs, lda, ldb);
      }
    }
  };
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
    SCOPED_TRACE(hwy::TargetName(target));
    hwy::SetSupportedTargetsForTest(target);
    // Exercises tight, asymmetric, negative, zero (broadcast), and 512, 1024,
    // and 4096 byte L1D set alignment branches across all vector widths.
    check_byte_strides(1, 64, {0, 64, -64, 128, 512, 1024, 4096, -4096},
                       {64, -64, 128, 512, -512, 1024, -1024, 4096});
    check_byte_strides(2, 32, {0, 64, -64, 128, 1024}, {64, -64, 128, 1024});
    check_byte_strides(4, 16, {0, 64, -64, 128, -1024, 4096},
                       {64, -64, 128, 1024});
    check_byte_strides(8, 8, {0, 64, -64, 128, 1024}, {64, -64, 128, 1024});
    check_byte_strides(16, 4, {0, 64, -64, 128, 1024}, {64, -64, 128, 1024});
    check_byte_strides(1, 32, {0, 32, -32, 96, -1024}, {32, -32, 96, 1024});
    check_byte_strides(2, 16, {0, 32, -32, 96, -1024}, {32, -32, 96, 1024});
    check_byte_strides(4, 8, {0, 32, -32, 96, -1024}, {32, -32, 96, 1024});
    check_byte_strides(8, 4, {0, 32, -32, 96, -1024}, {32, -32, 96, 1024});
    check_byte_strides(16, 2, {0, 32, -32, 96, 1024}, {32, -32, 96, 1024});
    check_byte_strides(1, 16, {0, 16, -16, 64, -128}, {16, -16, 64, 128});
    check_byte_strides(2, 8, {0, 16, -16, 64, -128}, {16, -16, 64, 128});
    check_byte_strides(4, 4, {0, 16, -16, 64, -128}, {16, -16, 64, 128});
    check_byte_strides(1, 8, {0, 8, -8, 64}, {8, -8, 64});
  }
}

// Same tile shapes as above, but transposing a macroblock of
// OuterBlockSize() x OuterBlockSize() tiles.
TEST(TransposeMacroKernelTest, ExactEquivalence) {
  const absl::Cleanup reset_targets = [] {
    hwy::SetSupportedTargetsForTest(0);
  };
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
    SCOPED_TRACE(hwy::TargetName(target));
    hwy::SetSupportedTargetsForTest(target);
    for (const KernelShape& shape : kKernelShapes) {
      CheckSquareKernel(shape.elem_size, shape.bs,
                        OuterBlockSize(shape.elem_size, shape.bs));
    }
  }
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
      [] { return GetTransposeMacroKernelOp(sizeof(uint16_t), 8); });
  expect_eager_init([] { return GetPackSubbyteOp<4>(); });
  expect_eager_init([] { return GetDeinterleaveOp(sizeof(uint32_t), 2); });
}

TEST(TransposeKernelsTest, MacroKernelUnsupportedReturnsNull) {
  EXPECT_EQ(GetTransposeMacroKernelOp(0, 4), nullptr);
  EXPECT_EQ(GetTransposeMacroKernelOp(3, 4), nullptr);
  EXPECT_EQ(GetTransposeMacroKernelOp(1, 1), nullptr);
  EXPECT_EQ(GetTransposeMacroKernelOp(1, 3), nullptr);
  EXPECT_EQ(GetTransposeMacroKernelOp(1, 128), nullptr);
  EXPECT_EQ(GetTransposeMacroKernelOp(16, 8), nullptr);
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
