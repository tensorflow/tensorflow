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
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "benchmark/benchmark.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/util/dynamic_bitset.h"

namespace xla {
namespace {

// Generate evenly spaced indices
std::vector<int64_t> GenerateIndices(int64_t n, int64_t max_val) {
  std::vector<int64_t> indices;
  indices.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    indices.push_back(i * (max_val / n));
  }
  return indices;
}

void BM_Add_DynamicBitset(benchmark::State& state) {
  int64_t n = state.range(0);
  std::vector<int64_t> indices = GenerateIndices(n, n * 2);
  for (auto _ : state) {
    DynamicBitset bitset;
    for (int64_t idx : indices) {
      bitset.Add(idx);
    }
    benchmark::DoNotOptimize(bitset);
  }
}
BENCHMARK(BM_Add_DynamicBitset)->Arg(32)->Arg(128)->Arg(512);

void BM_Add_FlatHashSet(benchmark::State& state) {
  int64_t n = state.range(0);
  std::vector<int64_t> indices = GenerateIndices(n, n * 2);
  for (auto _ : state) {
    absl::flat_hash_set<int64_t> s;
    for (int64_t idx : indices) {
      s.insert(idx);
    }
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_Add_FlatHashSet)->Arg(32)->Arg(128)->Arg(512);

void BM_Contains_DynamicBitset(benchmark::State& state) {
  int64_t n = state.range(0);
  std::vector<int64_t> indices = GenerateIndices(n, n * 2);
  DynamicBitset bitset;
  for (int64_t idx : indices) {
    bitset.Add(idx);
  }
  for (auto _ : state) {
    for (int64_t idx : indices) {
      benchmark::DoNotOptimize(bitset.Contains(idx));
    }
  }
}
BENCHMARK(BM_Contains_DynamicBitset)->Arg(32)->Arg(128)->Arg(512);

void BM_Contains_FlatHashSet(benchmark::State& state) {
  int64_t n = state.range(0);
  std::vector<int64_t> indices = GenerateIndices(n, n * 2);
  absl::flat_hash_set<int64_t> s;
  for (int64_t idx : indices) {
    s.insert(idx);
  }
  for (auto _ : state) {
    for (int64_t idx : indices) {
      benchmark::DoNotOptimize(s.contains(idx));
    }
  }
}
BENCHMARK(BM_Contains_FlatHashSet)->Arg(32)->Arg(128)->Arg(512);

void BM_Merge_DynamicBitset(benchmark::State& state) {
  int64_t n = state.range(0);
  std::vector<int64_t> indices1 = GenerateIndices(n, n * 2);
  std::vector<int64_t> indices2 = GenerateIndices(n, n * 3);
  DynamicBitset bitset1;
  for (int64_t idx : indices1) {
    bitset1.Add(idx);
  }
  DynamicBitset bitset2;
  for (int64_t idx : indices2) {
    bitset2.Add(idx);
  }

  for (auto _ : state) {
    DynamicBitset tmp = bitset1;
    tmp.Merge(bitset2);
    benchmark::DoNotOptimize(tmp);
  }
}
BENCHMARK(BM_Merge_DynamicBitset)->Arg(32)->Arg(128)->Arg(512);

void BM_Merge_FlatHashSet(benchmark::State& state) {
  int64_t n = state.range(0);
  std::vector<int64_t> indices1 = GenerateIndices(n, n * 2);
  std::vector<int64_t> indices2 = GenerateIndices(n, n * 3);
  absl::flat_hash_set<int64_t> s1;
  for (int64_t idx : indices1) {
    s1.insert(idx);
  }
  absl::flat_hash_set<int64_t> s2;
  for (int64_t idx : indices2) {
    s2.insert(idx);
  }

  for (auto _ : state) {
    absl::flat_hash_set<int64_t> tmp = s1;
    for (int64_t idx : s2) {  // NOLINT
      tmp.insert(idx);
    }
    benchmark::DoNotOptimize(tmp);
  }
}
BENCHMARK(BM_Merge_FlatHashSet)->Arg(32)->Arg(128)->Arg(512);

}  // namespace
}  // namespace xla
