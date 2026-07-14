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

// Benchmark targeting the MJX-like workload pattern from
// https://github.com/jax-ml/jax/issues/26021
//
// The MJX (MuJoCo JAX) robotics workload consists of many small matrix
// operations (kinematics, dynamics computations) on tensors sized for typical
// robots (29-36 DoF). This results in HLO programs with:
//   - Many thunks (50-200+) executed sequentially
//   - Each thunk operating on small buffers (< 512 bytes for scalars/vectors,
//     < 10KB for small matrices)
//   - Mix of kernel thunks (element-wise ops, dots) and custom call thunks
//   - Total execution time is very short (0.02-0.13ms), making per-thunk
//     overhead the dominant factor
//
// This benchmark profiles the overhead of:
//   1. ThunkExecutor sequential dispatch for many small ops
//   2. FFI custom call overhead (call frame pooling, context creation)
//   3. Small dot products and element-wise operations

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
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

//===----------------------------------------------------------------------===//
// Benchmark 1: Many sequential small ops (MJX-like kinematics pattern)
//
// This simulates the pattern where MJX computes kinematics chains:
// many sequential small matrix multiplications and additions.
//===----------------------------------------------------------------------===//

static void BM_ManySmallSequentialOps(benchmark::State& state,
                                      HloBenchmarkOptions options) {
  const int64_t num_ops = state.range(0);
  const int64_t n = 6;  // Typical spatial vector size in robotics

  // Build a chain of small additions: each depends on the previous.
  // This forces sequential execution in the thunk executor.
  std::string hlo = absl::StrFormat(R"(HloModule many_small_ops

ENTRY e {
  p0 = f64[%1$d,%1$d] parameter(0)
  p1 = f64[%1$d,%1$d] parameter(1)
  add0 = f64[%1$d,%1$d] add(p0, p1)
)",
                                    n);

  for (int64_t i = 1; i < num_ops; ++i) {
    absl::StrAppendFormat(&hlo, "  add%1$d = f64[%2$d,%2$d] add(add%3$d, p1)\n",
                          i, n, i - 1);
  }

  absl::StrAppendFormat(&hlo,
                        "  ROOT out = f64[%1$d,%1$d] add(add%2$d, p0)\n}\n", n,
                        num_ops - 1);

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F64, {n, n});
  auto p0 = LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  auto p1 = LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  CHECK_OK(p0);
  CHECK_OK(p1);

  std::vector<const Literal*> args = {&p0.value(), &p1.value()};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

//===----------------------------------------------------------------------===//
// Benchmark 2: Many small dot products (MJX mass matrix pattern)
//
// Simulates the pattern where MJX computes mass matrices through
// composite rigid body algorithm: many small matrix multiplications.
//===----------------------------------------------------------------------===//

static void BM_ManySmallDots(benchmark::State& state,
                             HloBenchmarkOptions options) {
  int64_t num_dots = state.range(0);
  int64_t n = 6;  // 6x6 spatial inertia matrices

  std::string hlo = absl::StrFormat(R"(HloModule many_small_dots

ENTRY e {
  p0 = f64[%1$d,%1$d] parameter(0)
  p1 = f64[%1$d,%1$d] parameter(1)
  dot0 = f64[%1$d,%1$d] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
)",
                                    n);

  for (int64_t i = 1; i < num_dots; ++i) {
    absl::StrAppendFormat(
        &hlo,
        "  dot%1$d = f64[%2$d,%2$d] dot(dot%3$d, p1), "
        "lhs_contracting_dims={1}, rhs_contracting_dims={0}\n",
        i, n, i - 1);
  }

  absl::StrAppendFormat(&hlo,
                        "  ROOT out = f64[%1$d,%1$d] add(dot%2$d, p0)\n}\n", n,
                        num_dots - 1);

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F64, {n, n});
  auto p0 = LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  auto p1 = LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  CHECK_OK(p0);
  CHECK_OK(p1);

  std::vector<const Literal*> args = {&p0.value(), &p1.value()};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

//===----------------------------------------------------------------------===//
// Benchmark 3: Mixed ops pattern (MJX gravity/RNE pattern)
//
// Simulates the recursive Newton-Euler pattern: mix of dots, adds,
// reshapes, and slices operating on small tensors.
//===----------------------------------------------------------------------===//

static void BM_MixedSmallOps(benchmark::State& state,
                             HloBenchmarkOptions options) {
  const int64_t num_joints = std::min(state.range(0), int64_t{6});

  // Simulates a simplified kinematic chain computation per joint:
  // For each joint: multiply transform, add bias, slice result
  std::string hlo = R"(HloModule mixed_small_ops

ENTRY e {
  p_q = f64[36] parameter(0)
  p_bias = f64[6,6] parameter(1)
  slice0 = f64[6] slice(p_q), slice={[0:6]}
  bcast0 = f64[6,6] broadcast(slice0), dimensions={0}
  add0 = f64[6,6] add(bcast0, p_bias)
)";

  for (int64_t i = 1; i < num_joints && i < 6; ++i) {
    int64_t start = i * 6;
    int64_t end = start + 6;
    absl::StrAppendFormat(
        &hlo, R"(  slice%1$d = f64[6] slice(p_q), slice={[%2$d:%3$d]}
  bcast%1$d = f64[6,6] broadcast(slice%1$d), dimensions={0}
  mul%1$d = f64[6,6] dot(add%4$d, bcast%1$d), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add%1$d = f64[6,6] add(mul%1$d, p_bias)
)",
        i, start, end, i - 1);
  }

  absl::StrAppendFormat(&hlo, "  ROOT out = f64[6,6] add(add%d, p_bias)\n}\n",
                        num_joints - 1);

  std::minstd_rand0 engine;
  auto q_shape = ShapeUtil::MakeShape(F64, {36});
  auto bias_shape = ShapeUtil::MakeShape(F64, {6, 6});
  auto p_q = LiteralUtil::CreateRandomLiteral<F64>(q_shape, &engine, 1.0, 0.1);
  auto p_bias =
      LiteralUtil::CreateRandomLiteral<F64>(bias_shape, &engine, 1.0, 0.1);
  CHECK_OK(p_q);
  CHECK_OK(p_bias);

  std::vector<const Literal*> args = {&p_q.value(), &p_bias.value()};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

//===----------------------------------------------------------------------===//
// Benchmark 4: FFI custom call overhead with varying buffer counts
//
// Tests the overhead of the FFI call path specifically, measuring:
// - ObjectPool call frame retrieval
// - Buffer address resolution
// - ExecutionContext creation
// - FFI handler dispatch
//===----------------------------------------------------------------------===//

static absl::Status NoopFFI2In1Out(
    ffi::Buffer<PrimitiveType::F64> arg0, ffi::Buffer<PrimitiveType::F64> arg1,
    ffi::Result<ffi::Buffer<PrimitiveType::F64>> ret0) {
  // Must avoid leaving output uninitialized as NaNs/denormals can severely
  // skew CPU performance metrics.
  std::fill_n(ret0->typed_data(), ret0->element_count(), 0.0);
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kNoopFFI2In1Out, NoopFFI2In1Out,
                       ffi::Ffi::Bind()
                           .Arg<ffi::Buffer<PrimitiveType::F64>>()
                           .Arg<ffi::Buffer<PrimitiveType::F64>>()
                           .Ret<ffi::Buffer<PrimitiveType::F64>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_bm$$noop_2in_1out", "Host",
                         kNoopFFI2In1Out);

static void BM_CustomCallChain(benchmark::State& state,
                               HloBenchmarkOptions options) {
  int64_t num_calls = state.range(0);
  int64_t n = 6;

  // Chain of custom calls, each depending on the previous output.
  // This directly measures per-custom-call overhead.
  std::string hlo = absl::StrFormat(R"(HloModule custom_call_chain

ENTRY e {
  p0 = f64[%1$d,%1$d] parameter(0)
  p1 = f64[%1$d,%1$d] parameter(1)
  cc0 = f64[%1$d,%1$d] custom-call(p0, p1), custom_call_target="__xla_bm$$noop_2in_1out", api_version=API_VERSION_TYPED_FFI
)",
                                    n);

  for (int64_t i = 1; i < num_calls; ++i) {
    absl::StrAppendFormat(&hlo,
                          "  cc%1$d = f64[%2$d,%2$d] custom-call(cc%3$d, p1), "
                          "custom_call_target=\"__xla_bm$$noop_2in_1out\", "
                          "api_version=API_VERSION_TYPED_FFI\n",
                          i, n, i - 1);
  }

  absl::StrAppendFormat(&hlo,
                        "  ROOT out = f64[%1$d,%1$d] add(cc%2$d, p0)\n}\n", n,
                        num_calls - 1);

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F64, {n, n});
  auto p0 = LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  auto p1 = LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  CHECK_OK(p0);
  CHECK_OK(p1);

  std::vector<const Literal*> args = {&p0.value(), &p1.value()};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

//===----------------------------------------------------------------------===//
// Benchmark 5: Interleaved compute and custom calls
//
// Simulates the real MJX pattern where compute kernels and FFI calls
// are interleaved (as in kinematics + dynamics computations).
//===----------------------------------------------------------------------===//

static void BM_InterleavedComputeAndCustomCalls(benchmark::State& state,
                                                HloBenchmarkOptions options) {
  const int64_t num_iterations = state.range(0);
  const int64_t n = 6;

  std::string hlo = absl::StrFormat(R"(HloModule interleaved

ENTRY e {
  p0 = f64[%1$d,%1$d] parameter(0)
  p1 = f64[%1$d,%1$d] parameter(1)
  dot0 = f64[%1$d,%1$d] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add0 = f64[%1$d,%1$d] add(dot0, p0)
  cc0 = f64[%1$d,%1$d] custom-call(add0, p1), custom_call_target="__xla_bm$$noop_2in_1out", api_version=API_VERSION_TYPED_FFI
)",
                                    n);

  for (int64_t i = 1; i < num_iterations; ++i) {
    absl::StrAppendFormat(
        &hlo,
        R"(  dot%1$d = f64[%2$d,%2$d] dot(cc%3$d, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add%1$d = f64[%2$d,%2$d] add(dot%1$d, cc%3$d)
  cc%1$d = f64[%2$d,%2$d] custom-call(add%1$d, p1), custom_call_target="__xla_bm$$noop_2in_1out", api_version=API_VERSION_TYPED_FFI
)",
        i, n, i - 1);
  }

  absl::StrAppendFormat(&hlo,
                        "  ROOT out = f64[%1$d,%1$d] add(cc%2$d, p0)\n}\n", n,
                        num_iterations - 1);

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F64, {n, n});
  auto p0 = LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  auto p1 = LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  CHECK_OK(p0);
  CHECK_OK(p1);

  std::vector<const Literal*> args = {&p0.value(), &p1.value()};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

//===----------------------------------------------------------------------===//
// Register benchmarks
//===----------------------------------------------------------------------===//

// Many sequential small element-wise ops (tests ThunkExecutor overhead)
XLA_CPU_BENCHMARK(BM_ManySmallSequentialOps)
    ->MeasureProcessCPUTime()
    ->Arg(10)
    ->Arg(25)
    ->Arg(50)
    ->Arg(100)
    ->Arg(200);

// Many small matrix multiplications (tests kernel dispatch overhead)
XLA_CPU_BENCHMARK(BM_ManySmallDots)
    ->MeasureProcessCPUTime()
    ->Arg(5)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50);

// Mixed ops simulating kinematic chain
XLA_CPU_BENCHMARK(BM_MixedSmallOps)
    ->MeasureProcessCPUTime()
    ->Arg(2)
    ->Arg(4)
    ->Arg(6);

// Custom call chain (tests FFI dispatch overhead specifically)
XLA_CPU_BENCHMARK(BM_CustomCallChain)
    ->MeasureProcessCPUTime()
    ->Arg(1)
    ->Arg(5)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50)
    ->Arg(100);

// Interleaved compute + custom calls (tests combined overhead)
XLA_CPU_BENCHMARK(BM_InterleavedComputeAndCustomCalls)
    ->MeasureProcessCPUTime()
    ->Arg(5)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50);

}  // namespace
}  // namespace xla::cpu
