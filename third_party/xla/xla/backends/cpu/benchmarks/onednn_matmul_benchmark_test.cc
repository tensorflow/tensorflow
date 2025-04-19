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

#if defined(INTEL_MKL)

#include <cstdint>
#include <random>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test.h"

namespace xla::cpu {

static void BM_oneDNN_MM(benchmark::State& state) {
  PrimitiveType dtype = static_cast<PrimitiveType>(state.range(0));
  int64_t d0 = state.range(1);

  absl::string_view hlo = R"(
    HloModule oneDNN_$dtype_$d0

    ENTRY e {
      lhs = $dtype[128,$d0] parameter(0)
      rhs = $dtype[$d0,256] parameter(1)
      ROOT custom-call = $dtype[128,256] custom-call(lhs, rhs), 
                  custom_call_target="__onednn$matmul"
    }
  )";

  std::minstd_rand0 engine;

  auto lhs_shape = ShapeUtil::MakeShape(dtype, {128, d0});
  auto rhs_shape = ShapeUtil::MakeShape(dtype, {d0, 256});
  Literal p0, p1;

  if (dtype == F32) {
    p0 = *LiteralUtil::CreateRandomLiteral<F32>(lhs_shape, &engine, 1.0f, 0.1f);
    p1 = *LiteralUtil::CreateRandomLiteral<F32>(rhs_shape, &engine, 1.0f, 0.1f);
  } else if (dtype == BF16 && IsSupportedType(BF16)) {
    p0 =
        *LiteralUtil::CreateRandomLiteral<BF16>(lhs_shape, &engine, 1.0f, 0.1f);
    p1 =
        *LiteralUtil::CreateRandomLiteral<BF16>(rhs_shape, &engine, 1.0f, 0.1f);
  } else if (dtype == F16 && IsSupportedType(F16)) {
    p0 = *LiteralUtil::CreateRandomLiteral<F16>(lhs_shape, &engine, 1.0f, 0.1f);
    p1 = *LiteralUtil::CreateRandomLiteral<F16>(rhs_shape, &engine, 1.0f, 0.1f);
  } else {
    VLOG(0) << primitive_util::LowercasePrimitiveTypeName(dtype)
            << " not supported on this platform";
    return;
  }

  std::vector<const Literal*> args = {&p0, &p1};
  HloBenchmarkOptions benchmark_options;
  benchmark_options.use_thunk_runtime = false;
  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$dtype", primitive_util::LowercasePrimitiveTypeName(dtype)},
       {"$d0", absl::StrCat(d0)}},
      benchmark_options));
}

#define BENCHMARK_ONEDNN_MM(dtype) \
  BENCHMARK(BM_oneDNN_MM)          \
      ->MeasureProcessCPUTime()    \
      ->Args({dtype, 512})         \
      ->Args({dtype, 1024})        \
      ->Args({dtype, 2048})

BENCHMARK_ONEDNN_MM(F32);
BENCHMARK_ONEDNN_MM(BF16);
BENCHMARK_ONEDNN_MM(F16);

}  // namespace xla::cpu

#endif  // INTEL_MKL
