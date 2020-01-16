/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include <cstdio>
#include <cstdlib>
#include <string>

#include "tensorflow/lite/experimental/ruy/test.h"

namespace ruy {

using LhsScalar = RUY_TEST_LHSSCALAR;
using RhsScalar = RUY_TEST_RHSSCALAR;
using AccumScalar = RUY_TEST_ACCUMSCALAR;
using DstScalar = RUY_TEST_DSTSCALAR;
using TestSetType =
    TestSet<LhsScalar, RhsScalar, BasicSpec<AccumScalar, DstScalar>>;

struct BenchmarkShape {
  int rows;
  int depth;
  int cols;
  int symm_lhs;
  int symm_rhs;
};

template <typename TestSetType>
std::vector<std::unique_ptr<TestResult<DstScalar>>> BenchmarkRCC(
    const BenchmarkShape& shape) {
  TestSetType test_set;
  test_set.rows = shape.rows;
  test_set.depth = shape.depth;
  test_set.cols = shape.cols;
  test_set.lhs_order = Order::kRowMajor;
  test_set.rhs_order = Order::kColMajor;
  test_set.dst_order = Order::kColMajor;
  test_set.layout_style = LayoutStyle::kPackedLinear;
  test_set.benchmark = true;
  const int asymmetry_lhs = shape.symm_lhs ? 0 : 1;
  const int asymmetry_rhs = shape.symm_rhs ? 0 : 1;
  test_set.lhs_zero_point = SymmetricZeroPoint<LhsScalar>() + asymmetry_lhs;
  test_set.rhs_zero_point = SymmetricZeroPoint<RhsScalar>() + asymmetry_rhs;
  test_set.use_specified_zero_points = true;
  test_set.perchannel = GetBoolEnvVarOrFalse("PERCHANNEL");
  test_set.benchmark_prepack_lhs = GetBoolEnvVarOrFalse("PREPACK_LHS");
  test_set.benchmark_prepack_rhs = GetBoolEnvVarOrFalse("PREPACK_RHS");
  test_set.Run();
  return std::move(test_set.results);
}

void Benchmark() {
  const bool symm_lhs = std::is_floating_point<LhsScalar>::value ||
                        GetBoolEnvVarOrFalse("SYMM_LHS");
  const bool symm_rhs = std::is_floating_point<RhsScalar>::value ||
                        GetBoolEnvVarOrFalse("SYMM_RHS");
  const bool benchmark_cubic = GetBoolEnvVarOrFalse("RUY_BENCHMARK_CUBIC");
  const int explicit_rows = GetIntEnvVarOrZero("ROWS");
  const int explicit_cols = GetIntEnvVarOrZero("COLS");
  const int explicit_depth = GetIntEnvVarOrZero("DEPTH");

  std::vector<BenchmarkShape> shapes;

  // Often 8 is used for this multiplier, but to check teeny sizes one can
  // use 1.
  static constexpr int cubic_size_multiplier = 8;

  if (benchmark_cubic) {
    std::vector<int> sizes;
    for (int i = 2 * cubic_size_multiplier; i <= (512 * cubic_size_multiplier);
         i *= 2) {
      sizes.push_back(i);
      if (i < (512 * cubic_size_multiplier)) {
        sizes.push_back(i * 3 / 2);
      }
    }
    for (int i : sizes) {
      BenchmarkShape shape;
      // Even in cubic mode, one may still override an individual dimension
      // to allow testing a batch of rectangular sizes.
      shape.rows = explicit_rows ? explicit_rows : i;
      shape.cols = explicit_cols ? explicit_cols : i;
      shape.depth = explicit_depth ? explicit_depth : i;
      shape.symm_lhs = symm_lhs;
      shape.symm_rhs = symm_rhs;
      shapes.push_back(shape);
    }
  } else {
    BenchmarkShape shape;
    shape.rows = explicit_rows;
    shape.cols = explicit_cols;
    shape.depth = explicit_depth;
    if (!shape.rows || !shape.depth || !shape.cols) {
      fprintf(stderr,
              "Please specify positive sizes with these env vars: ROWS, DEPTH, "
              "COLS.\n");
      exit(1);
    }
    shape.symm_lhs = symm_lhs;
    shape.symm_rhs = symm_rhs;
    shapes.push_back(shape);
  }

  for (int i = 0; i < shapes.size(); i++) {
    const auto& shape = shapes[i];
    const auto& results = BenchmarkRCC<TestSetType>(shape);
    if (i == 0) {
      if (benchmark_cubic) {
        printf("size");
        for (const auto& result : results) {
          printf(",%s", PathName(*result).c_str());
        }
        printf("\n");
      } else {
        printf("path,shape,Gop/s\n");
      }
      fflush(stdout);
    }
    if (benchmark_cubic) {
      printf("%d", shape.rows);
      for (const auto& result : results) {
        printf(",%.4g", 2.0e-9 * shape.rows * shape.cols * shape.depth /
                            result->latency);
        if (GetBoolEnvVarOrFalse("RUY_BENCHMARK_PMU")) {
          printf(",%.3g,%.3g,%.3g,%.3g,%.3g,%.3g,%.3g,%.3g",
                 result->l1_refill_rate, result->l2_refill_rate,
                 result->l3_refill_rate, result->l1tlb_refill_rate,
                 result->l2tlb_refill_rate, result->mispred_rate,
                 result->frontend_stall_rate, result->backend_stall_rate);
        }
      }
      printf("\n");
      fflush(stdout);
    } else {
      for (const auto& result : results) {
        printf(
            "%s,%dx%dx%d,%.4g", PathName(*result).c_str(), shape.rows,
            shape.depth, shape.cols,
            2.0e-9 * shape.rows * shape.cols * shape.depth / result->latency);
        if (GetBoolEnvVarOrFalse("RUY_BENCHMARK_PMU")) {
          printf(",%.3g,%.3g,%.3g,%.3g,%.3g,%.3g,%.3g,%.3g",
                 result->l1_refill_rate, result->l2_refill_rate,
                 result->l3_refill_rate, result->l1tlb_refill_rate,
                 result->l2tlb_refill_rate, result->mispred_rate,
                 result->frontend_stall_rate, result->backend_stall_rate);
        }
        printf("\n");
      }
      fflush(stdout);
    }
  }
}

}  // namespace ruy

int main() { ruy::Benchmark(); }
