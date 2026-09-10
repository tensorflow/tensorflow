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

#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/stacktrace_handler.h"

namespace xla {
namespace {

template <PrimitiveType type>
void BM_MakeFakeLiteral(benchmark::State& state) {
  int64_t num_elements = state.range(0);
  Shape shape = ShapeUtil::MakeShape(type, {num_elements});
  for (auto _ : state) {
    auto literal_or = MakeFakeLiteral(shape, /*pseudo_random=*/true);
    CHECK_OK(literal_or.status());
    benchmark::DoNotOptimize(literal_or);
  }
}

BENCHMARK_TEMPLATE(BM_MakeFakeLiteral, F32)->Arg(1200000000)->Iterations(1);
BENCHMARK_TEMPLATE(BM_MakeFakeLiteral, F16)->Arg(1200000000)->Iterations(1);
BENCHMARK_TEMPLATE(BM_MakeFakeLiteral, BF16)->Arg(1200000000)->Iterations(1);
BENCHMARK_TEMPLATE(BM_MakeFakeLiteral, F8E5M2)->Arg(1200000000)->Iterations(1);
BENCHMARK_TEMPLATE(BM_MakeFakeLiteral, F8E4M3FN)
    ->Arg(1200000000)
    ->Iterations(1);
BENCHMARK_TEMPLATE(BM_MakeFakeLiteral, F4E2M1FN)
    ->Arg(1200000000)
    ->Iterations(1);

void RunPrimitiveTypeSwitchBenchmark(benchmark::State& state,
                                     PrimitiveType type) {
  int64_t count = 0;
  for (auto _ : state) {
    primitive_util::PrimitiveTypeSwitch<void>(
        [&](auto primitive_type_constant) {
          count += decltype(primitive_type_constant)::value;
        },
        type);
  }
  benchmark::DoNotOptimize(count);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  using xla::BF16;
  using xla::F32;
  using xla::F4E2M1FN;
  using xla::PrimitiveType;
  using xla::RunPrimitiveTypeSwitchBenchmark;
  using xla::primitive_util::LowercasePrimitiveTypeName;
  tsl::testing::InstallStacktraceHandler();

  ::benchmark::Initialize(&argc, argv);
  testing::InitGoogleTest(&argc, argv);

  for (PrimitiveType type : {F32, BF16, F4E2M1FN}) {
    std::string name = absl::StrCat("BM_PrimitiveTypeSwitch/",
                                    LowercasePrimitiveTypeName(type));
    benchmark::RegisterBenchmark(name.c_str(), [type](benchmark::State& state) {
      RunPrimitiveTypeSwitchBenchmark(state, type);
    });
  }

  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
