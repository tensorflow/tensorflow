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
#include <random>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/ffi/ffi.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

// Copied from third_party/py/jax/jaxlib/cpu/lapack_kernels.h so we don't have
// to include jaxlib headers into XLA.
enum class ComputationMode : char {
  kComputeFullUVt = 'A',  // Compute U and VT
  kComputeMinUVt = 'S',   // Compute min(M, N) columns of U and rows of VT
  kComputeVtOverwriteXPartialU = 'O',  // Compute VT, overwrite X
                                       // with partial U
  kNoComputeUVt = 'N',                 // Do not compute U or VT
};

static void BM_CustomCall_SVD(benchmark::State& state,
                              HloBenchmarkOptions options) {
  const char* hlo = R"(
    HloModule module

    ENTRY custom_call {
      %Arg_0.1 = f32[$batch_size,$d,$d]{2,1,0} parameter(0)
      ROOT %svd.16 = (f32[$batch_size,$d,$d]{1,2,0}, f32[$batch_size,$d]{1,0}, f32[$batch_size,$d,$d]{1,2,0}, f32[$batch_size,$d,$d]{1,2,0}, s32[$batch_size]{0}) custom-call(%Arg_0.1), custom_call_target="lapack_sgesdd_ffi", operand_layout_constraints={f32[$batch_size,$d,$d]{1,2,0}}, output_to_operand_aliasing={{0}: (0, {})}, api_version=API_VERSION_TYPED_FFI, frontend_attributes={num_batch_dims="1",xla.sdy.sharding_rule="#sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=$batch_size, j=$d, k=$d, l=$d, m=$d, n=$d, o=$d, p=$d, q=$d, r=$d}, custom>"}, metadata={op_name="svd" stack_frame_id=6}, backend_config={mode = $mode : ui8}
    }
  )";

  std::minstd_rand0 engine;

  int64_t batch_size = state.range(0);
  int64_t d = state.range(1);
  int64_t mode = state.range(2);

  auto shape = ShapeUtil::MakeShape(F32, {batch_size, d, d});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args(1, &p0);

  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {{"$batch_size", absl::StrCat(batch_size)},
                            {"$d", absl::StrCat(d)},
                            {"$mode", absl::StrCat(mode)}},
                           options));
}

// Currently testing just ComputeFullUVt as it is the most complex case.
XLA_CPU_BENCHMARK(BM_CustomCall_SVD)
    ->ArgNames({"batch_size", "d", "mode"})
    // 2x2
    ->Args({1, 2, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({2, 2, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({4, 2, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({4096, 2, static_cast<int>(ComputationMode::kComputeFullUVt)})

    // 8x8
    ->Args({1, 8, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({16, 8, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({64, 8, static_cast<int>(ComputationMode::kComputeFullUVt)})

    // 16x16
    ->Args({1, 16, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({16, 16, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({64, 16, static_cast<int>(ComputationMode::kComputeFullUVt)})

    // 64x64
    ->Args({1, 64, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({16, 64, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({64, 64, static_cast<int>(ComputationMode::kComputeFullUVt)})

    // 128x128
    ->Args({1, 128, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({16, 128, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({64, 128, static_cast<int>(ComputationMode::kComputeFullUVt)})

    // 1024x1024
    ->Args({1, 1024, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({16, 1024, static_cast<int>(ComputationMode::kComputeFullUVt)})
    ->Args({64, 1024, static_cast<int>(ComputationMode::kComputeFullUVt)})

    ->MeasureProcessCPUTime();

}  // namespace

}  // namespace xla::cpu
