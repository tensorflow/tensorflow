/* Copyright 2024 The OpenXLA Authors.

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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Target/TargetMachine.h"
#include "third_party/py/torch/aten/src/ATen/ATen.h"  // IWYU pragma: keep
#include "third_party/py/torch/aten/src/ATen/NativeFunctions.h"  // IWYU pragma: keep
#include "third_party/py/torch/aten/src/ATen/core/ATen_fwd.h"
#include "third_party/py/torch/aten/src/ATen/core/TensorBody.h"
#include "third_party/py/torch/aten/src/ATen/ops/rand.h"
#include "third_party/py/torch/aten/src/ATen/ops/tanh.h"
#include "third_party/py/torch/aten/src/ATen/ops/zeros_like.h"
#include "third_party/py/torch/c10/core/DeviceType.h"
#include "third_party/py/torch/c10/core/ScalarType.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/tanh.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

using ::xla::codegen::intrinsic::JitRunner;
using ::xla::codegen::intrinsics::Type;

static void BM_TanhF32(benchmark::State& state, HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule tanh_f32_$d0

    ENTRY e {
      input = f32[$d0] parameter(0)
      ROOT output = tanh(input)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F32, {d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_TanhF16(benchmark::State& state) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule tanh_f16_$d0

    ENTRY e {
      input = f16[$d0] parameter(0)
      ROOT output = tanh(input)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F16, {d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F16>(input_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

static void BM_TanhF64(benchmark::State& state, HloBenchmarkOptions options) {
  const int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule tanh_f64_$d0

    ENTRY e {
      input = f64[$d0] parameter(0)
      ROOT output = tanh(input)
    }
  )";

  std::minstd_rand0 engine;

  auto input_shape = ShapeUtil::MakeShape(F64, {d0});
  auto p0 =
      *LiteralUtil::CreateRandomLiteral<F64>(input_shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));

  state.SetItemsProcessed(state.iterations() * d0);
  state.SetBytesProcessed(state.iterations() * d0 * sizeof(double));
}

static void BM_TanhF64PytorchAten(benchmark::State& state) {
  const int64_t d0 = state.range(0);
  const size_t inner_loop = 10;
  CHECK_GT(d0, 0);
  CHECK_LT(d0, 1024 * 1024 * 1024);

  // Avoid ambiguity with single element constructor.
  std::vector<int64_t> tensor_size_vec = {d0};
  const at::IntArrayRef tensor_size{tensor_size_vec};
  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
  at::Tensor input = (at::rand(tensor_size, options) * 1.8) - 0.9;
  at::Tensor result = at::zeros_like(input);

  for (auto _ : state) {
    for (int i = 0; i < inner_loop; ++i) {
      benchmark::DoNotOptimize(input);
      at::tanh_out(result, input);
    }
  }
  benchmark::DoNotOptimize(result);
  state.SetItemsProcessed(state.iterations() * inner_loop * d0);
  state.SetBytesProcessed(state.iterations() * inner_loop * d0 *
                          sizeof(double));
}

std::tuple<JitRunner, std::string> CreateJitRunner(
    Type type, std::optional<int> unroll_factor) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Function* tanh_func =
      ::xla::codegen::intrinsics::Tanh::CreateDefinition(module.get(), type)
          .value();
  tanh_func->setLinkage(llvm::Function::ExternalLinkage);
  llvm::verifyFunction(*tanh_func);
  std::string function_name = tanh_func->getName().str();
  if (unroll_factor.has_value()) {
    llvm::Function* k_times_wrapper = codegen::intrinsic::CreateKTimesWrapper(
        module.get(), tanh_func, *unroll_factor,
        type.vector_width().value_or(1));
    k_times_wrapper->setLinkage(llvm::Function::ExternalLinkage);
    llvm::verifyFunction(*k_times_wrapper);
    function_name = k_times_wrapper->getName().str();
  }
  return {JitRunner(std::move(module), std::move(context)), function_name};
}

template <size_t vector_width, PrimitiveType type, bool unroll>
static void BM_TanhF64XLAJitRunner(benchmark::State& state) {
  using NativeType = typename primitive_util::PrimitiveTypeToNative<type>::type;
  const int unroll_factor = state.range(0);
  const int iteration_count = state.range(1);
  auto [jit, function_name] = CreateJitRunner(
      Type::V(F64, 4),
      unroll ? std::make_optional(unroll_factor) : std::nullopt);
  auto fn = jit.GetVectorizedFn<vector_width, NativeType, NativeType>(
      function_name, iteration_count);

  std::array<NativeType, vector_width> vec;
  for (size_t i = 0; i < vector_width; ++i) {
    vec[i] = static_cast<NativeType>(i * 0.01);
  }
  std::array<NativeType, vector_width> result;
  for (auto s : state) {
    benchmark::DoNotOptimize(vec);
    result = fn(vec);
  }
  benchmark::DoNotOptimize(result);

  const size_t items_processed = state.iterations() * iteration_count *
                                 (unroll ? unroll_factor : 1) * vector_width;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(NativeType));
}

#define REGISTER_TANH_BENCHMARK(NAME) \
  XLA_CPU_BENCHMARK(NAME)             \
      ->MeasureProcessCPUTime()       \
      ->Arg(128)                      \
      ->Arg(256)                      \
      ->Arg(512)                      \
      ->Arg(1024)                     \
      ->Arg(4096);

REGISTER_TANH_BENCHMARK(BM_TanhF32);
REGISTER_TANH_BENCHMARK(BM_TanhF64);

// TODO(b/406431945): add AOT for f16 tanh
BENCHMARK(BM_TanhF16)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(4096);

BENCHMARK(BM_TanhF64PytorchAten)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(4096);

// BENCHMARK(BM_TanhF64XLAJitRunner<4, F64>)
//     ->MeasureProcessCPUTime()
//     ->ArgPair(10, 128)
//     ->ArgPair(10, 512)
//     ->ArgPair(10, 4096);
BENCHMARK(BM_TanhF64XLAJitRunner<8, F64, true>)
    ->MeasureProcessCPUTime()
    ->ArgPair(10, 128)
    ->ArgPair(10, 512)
    ->ArgPair(10, 4096)
    ->ArgPair(20, 4096);
BENCHMARK(BM_TanhF64XLAJitRunner<8, F64, false>)
    ->MeasureProcessCPUTime()
    ->ArgPair(10, 128)
    ->ArgPair(10, 512)
    ->ArgPair(10, 4096)
    ->ArgPair(20, 4096);
}  // namespace xla::cpu
