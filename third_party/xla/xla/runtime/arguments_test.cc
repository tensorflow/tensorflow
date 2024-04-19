/*
 * Copyright 2022 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xla/runtime/arguments.h"

#include <array>
#include <type_traits>
#include <utility>

#include "tsl/platform/test_benchmark.h"

namespace xla {
namespace runtime {

//===----------------------------------------------------------------------===//
// Benchmarks for constructing MemrefDesc.
//===----------------------------------------------------------------------===//

static void BM_CreateMemrefDesc_1d(benchmark::State& state) {
  void* ptr = reinterpret_cast<void*>(0xDEADBEEF);
  int64_t size = 123;
  int64_t stride = 456;

  int64_t num_memrefs = state.range(0);

  for (auto _ : state) {
    Arguments<MemrefDesc> memrefs(num_memrefs);

    for (unsigned i = 0; i < num_memrefs; ++i) {
      std::array<int64_t, 1> sizes = {size};
      std::array<int64_t, 1> strides = {stride};
      memrefs.emplace_back(PrimitiveType::S8, ptr, 0, sizes, strides);
    }

    benchmark::DoNotOptimize(memrefs);
  }
}

BENCHMARK(BM_CreateMemrefDesc_1d)->Arg(1)->Arg(4)->Arg(8)->Arg(12)->Arg(16);

//===----------------------------------------------------------------------===//
// Run benchmarks for verifying operands.
//===----------------------------------------------------------------------===//

static MemrefDesc GetFakeMemref(absl::Span<const int64_t> sizes) {
  return MemrefDesc(PrimitiveType::F32, nullptr, 0, sizes,
                    sizes /* fake strides*/);
}

static void BenchmarkVerifyMemrefOperand(benchmark::State& state,
                                         const MemrefDesc& memref) {
  MemrefType type(memref.sizes(), memref.dtype());

  for (auto _ : state) {
    if (auto st = VerifyMemrefArgument(0, type, memref); !st.ok()) break;
  }
}

static void BM_VerifyMemref_1d(benchmark::State& state) {
  auto memref = GetFakeMemref({1});
  BenchmarkVerifyMemrefOperand(state, memref);
}

static void BM_VerifyMemref_2d(benchmark::State& state) {
  auto memref = GetFakeMemref({1, 2});
  BenchmarkVerifyMemrefOperand(state, memref);
}

static void BM_VerifyMemref_3d(benchmark::State& state) {
  auto memref = GetFakeMemref({1, 2, 3});
  BenchmarkVerifyMemrefOperand(state, memref);
}

static void BM_VerifyMemref_4d(benchmark::State& state) {
  auto memref = GetFakeMemref({1, 2, 3, 4});
  BenchmarkVerifyMemrefOperand(state, memref);
}

static void BM_VerifyMemref_5d(benchmark::State& state) {
  auto memref = GetFakeMemref({1, 2, 3, 4, 5});
  BenchmarkVerifyMemrefOperand(state, memref);
}

BENCHMARK(BM_VerifyMemref_1d);
BENCHMARK(BM_VerifyMemref_2d);
BENCHMARK(BM_VerifyMemref_3d);
BENCHMARK(BM_VerifyMemref_4d);
BENCHMARK(BM_VerifyMemref_5d);

}  // namespace runtime
}  // namespace xla
