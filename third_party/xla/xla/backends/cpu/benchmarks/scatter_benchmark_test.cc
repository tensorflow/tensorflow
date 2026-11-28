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

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/transforms/expanders/scatter_simplifier.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

namespace {

constexpr std::minstd_rand0::result_type kBenchmarkSeed = 12345u;

enum class ScatterReducerKind {
  kAdd,
  kOverwrite,
};

// Generate random indices to be used in the scatter.
// To make them unique, we randomly select from the range [0, domain_size)
// by first shuffling the full range and then taking the first scatter_size
// elements.
Literal createR1ScatterIndices(int64_t domain_size, int64_t scatter_size,
                               std::minstd_rand0& engine) {
  Literal scatter_indices;

  std::vector<int32_t> scatter_indices_vector(domain_size);
  absl::c_iota(scatter_indices_vector, 0);
  std::shuffle(scatter_indices_vector.begin(), scatter_indices_vector.end(),
               engine);
  scatter_indices_vector.resize(scatter_size);
  scatter_indices = LiteralUtil::CreateR1<int32_t>(scatter_indices_vector);

  return scatter_indices;
}

// Creates a 2D literal to reduce from num_elems * step to num_elems elements.
Literal CreateReduceIndices(int32_t num_elems, int32_t step) {
  CHECK_GE(num_elems, step);
  CHECK_EQ(num_elems % step, 0);
  Array2D<int32_t> array(num_elems * step, 1);
  for (int i = 0; i < num_elems; ++i) {
    for (int j = 0; j < step; ++j) {
      array(i * step + j, 0) = i;
    }
  }
  return LiteralUtil::CreateR2FromArray2D(array);
}

// Generates distinct indices in [0, domain_size) with shape [num_updates, 1].
Literal CreateUniqueR2Indices(int64_t domain_size, int64_t num_updates,
                              std::minstd_rand0& engine) {
  CHECK_GT(domain_size, 0);
  CHECK_GE(num_updates, 0);
  CHECK_LE(num_updates, domain_size);
  std::vector<int32_t> pool(domain_size);
  absl::c_iota(pool, 0);
  for (int64_t i = 0; i < num_updates; ++i) {
    std::uniform_int_distribution<int64_t> dist(i, domain_size - 1);
    std::swap(pool[i], pool[dist(engine)]);
  }
  Literal indices(ShapeUtil::MakeShape(S32, {num_updates, 1}));
  absl::Span<int32_t> dst = indices.data<int32_t>();
  for (int64_t i = 0; i < num_updates; ++i) {
    dst[i] = pool[i];
  }
  return indices;
}

// Generates uniform random indices in [0, domain_size) shaped [num_updates, 1].
Literal CreateRandomR2Indices(int64_t domain_size, int64_t num_updates,
                              std::minstd_rand0& engine) {
  CHECK_GT(domain_size, 0);
  CHECK_GE(num_updates, 0);
  Literal indices(ShapeUtil::MakeShape(S32, {num_updates, 1}));
  absl::Span<int32_t> dst = indices.data<int32_t>();
  if (domain_size == 1) {
    absl::c_fill(dst, 0);
    return indices;
  }
  std::uniform_int_distribution<int32_t> dist(
      0, static_cast<int32_t>(domain_size - 1));
  for (int32_t& idx : dst) {
    idx = dist(engine);
  }
  return indices;
}

struct BalancedUpdates {
  Literal primary;
  std::optional<Literal> negated;
};

// Populates non zero signed updates paired per destination slot (+v then -v, or
// -v then +v) so within iteration accumulation stays in [-v, +v] for every
// slot and every update modifies stored bits. When `alternate_negated` is true
// and any slot receives an odd number of updates (including single hit bins in
// sparse histograms), also builds an elementwise negated literal swapped in on
// odd executions so every slot cancels to 0 every two iterations and never
// drifts into low precision float saturation.
template <typename NativeT>
BalancedUpdates CreateBalancedUpdates(PrimitiveType type,
                                      absl::Span<const int64_t> dims,
                                      int64_t num_slots,
                                      absl::Span<const int32_t> slot_indices,
                                      NativeT unit_val, bool alternate_negated,
                                      std::minstd_rand0& engine) {
  const int64_t num_updates = dims.front();
  CHECK_EQ(static_cast<int64_t>(slot_indices.size()), num_updates);
  CHECK_GT(num_slots, 0);
  const int64_t slice_elems = Product(dims.subspan(1));
  const NativeT neg_val = static_cast<NativeT>(-unit_val);

  // Per slot pending sign: 0 = balanced so far, 1 = emit +v next, -1 = emit -v
  // next.
  std::vector<int8_t> pending_sign(num_slots, 0);
  std::bernoulli_distribution flip_dist(0.5);

  const Shape shape = ShapeUtil::MakeShape(type, dims);
  Literal primary(shape);
  absl::Span<NativeT> primary_data = primary.data<NativeT>();
  for (int64_t u = 0; u < num_updates; ++u) {
    const int32_t slot = slot_indices[u];
    CHECK_GE(slot, 0);
    CHECK_LT(slot, num_slots);
    int8_t& pending = pending_sign[slot];
    bool use_neg = false;
    if (pending != 0) {
      use_neg = pending < 0;
      pending = 0;
    } else {
      use_neg = flip_dist(engine);
      pending = use_neg ? 1 : -1;
    }
    absl::Span<NativeT> dst =
        primary_data.subspan(u * slice_elems, slice_elems);
    absl::c_fill(dst, use_neg ? neg_val : unit_val);
  }

  std::optional<Literal> negated;
  if (alternate_negated &&
      absl::c_any_of(pending_sign, [](int8_t s) { return s != 0; })) {
    negated.emplace(shape);
    absl::Span<NativeT> negated_data = negated->data<NativeT>();
    for (int64_t i = 0; i < static_cast<int64_t>(primary_data.size()); ++i) {
      negated_data[i] = static_cast<NativeT>(-primary_data[i]);
    }
  }
  return {std::move(primary), std::move(negated)};
}

// Generates a 1D floating point literal uniformly distributed in [0.0, 1.0).
template <typename FloatT>
Literal CreateUniform01Literal(PrimitiveType type, int64_t num_elements,
                               std::minstd_rand0& engine) {
  Literal literal(ShapeUtil::MakeShape(type, {num_elements}));
  absl::Span<FloatT> data = literal.data<FloatT>();
  std::uniform_real_distribution<FloatT> dist(FloatT{0.0}, FloatT{1.0});
  for (FloatT& val : data) {
    val = dist(engine);
  }
  return literal;
}

template <typename NativeT>
void RunR1ScatterBenchmark(benchmark::State& state,
                           HloBenchmarkOptions& options, PrimitiveType type,
                           absl::string_view hlo_type, int64_t n_bins,
                           int64_t num_updates, bool unique_indices,
                           ScatterReducerKind reducer, NativeT unit_update) {
  constexpr absl::string_view kHlo = R"(
    HloModule bm_scatter_r1_$reducer_name, input_output_alias={ {}: (0, {}) }

    reducer {
      lhs = $type[] parameter(0)
      $reducer_body
    }

    ENTRY main {
      operand = $type[$n_bins] parameter(0)
      indices = s32[$num_updates,1] parameter(1)
      updates = $type[$num_updates] parameter(2)
      ROOT scatter = $type[$n_bins] scatter(operand, indices, updates),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=$unique_indices,
          to_apply=reducer
    }
  )";

  std::minstd_rand0 engine(kBenchmarkSeed);
  const Literal operand = LiteralUtil::CreateFromDimensions(type, {n_bins});
  const Literal indices =
      unique_indices ? CreateUniqueR2Indices(n_bins, num_updates, engine)
                     : CreateRandomR2Indices(n_bins, num_updates, engine);
  const BalancedUpdates updates = CreateBalancedUpdates<NativeT>(
      type, {num_updates}, n_bins, indices.data<int32_t>(), unit_update,
      /*alternate_negated=*/reducer == ScatterReducerKind::kAdd, engine);

  const std::string n_bins_str = absl::StrCat(n_bins);
  const std::string num_updates_str = absl::StrCat(num_updates);
  const absl::string_view unique_str = unique_indices ? "true" : "false";
  const absl::string_view reducer_name =
      reducer == ScatterReducerKind::kAdd ? "add" : "overwrite";
  const std::string reducer_body =
      reducer == ScatterReducerKind::kAdd
          ? absl::StrCat("rhs = ", hlo_type,
                         "[] parameter(1)\n      ROOT sum = ", hlo_type,
                         "[] add(lhs, rhs)")
          : absl::StrCat("ROOT rhs = ", hlo_type, "[] parameter(1)");
  const std::array<const Literal*, 3> args = {&operand, &indices,
                                              &updates.primary};
  options.alternate_odd_args.clear();
  if (updates.negated.has_value()) {
    options.alternate_odd_args = {nullptr, nullptr, &*updates.negated};
  }
  CHECK_OK(RunHloBenchmark(state, kHlo, args,
                           {{"$reducer_name", reducer_name},
                            {"$reducer_body", reducer_body},
                            {"$type", hlo_type},
                            {"$n_bins", n_bins_str},
                            {"$num_updates", num_updates_str},
                            {"$unique_indices", unique_str}},
                           options));
  options.alternate_odd_args.clear();
}

void RunSliceScatterF32Benchmark(benchmark::State& state,
                                 HloBenchmarkOptions& options,
                                 bool unique_indices,
                                 ScatterReducerKind reducer) {
  const int64_t num_rows = state.range(0);
  const int64_t slice_size = state.range(1);
  const int64_t num_updates = state.range(2);

  constexpr absl::string_view kHlo = R"(
    HloModule bm_scatter_slice_$reducer_name_f32, input_output_alias={ {}: (0, {}) }

    reducer {
      lhs = f32[] parameter(0)
      $reducer_body
    }

    ENTRY main {
      operand = f32[$num_rows,$slice_size] parameter(0)
      indices = s32[$num_updates,1] parameter(1)
      updates = f32[$num_updates,1,$slice_size] parameter(2)
      ROOT scatter = f32[$num_rows,$slice_size]
          scatter(operand, indices, updates),
          update_window_dims={1,2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=$unique_indices,
          to_apply=reducer
    }
  )";

  std::minstd_rand0 engine(kBenchmarkSeed);
  const Literal operand =
      LiteralUtil::CreateFromDimensions(F32, {num_rows, slice_size});
  const Literal indices =
      unique_indices ? CreateUniqueR2Indices(num_rows, num_updates, engine)
                     : CreateRandomR2Indices(num_rows, num_updates, engine);
  const BalancedUpdates updates = CreateBalancedUpdates<float>(
      F32, {num_updates, 1, slice_size}, num_rows, indices.data<int32_t>(),
      1.0f, /*alternate_negated=*/reducer == ScatterReducerKind::kAdd, engine);

  const std::string num_rows_str = absl::StrCat(num_rows);
  const std::string slice_size_str = absl::StrCat(slice_size);
  const std::string num_updates_str = absl::StrCat(num_updates);
  const absl::string_view unique_str = unique_indices ? "true" : "false";
  const absl::string_view reducer_name =
      reducer == ScatterReducerKind::kAdd ? "add" : "overwrite";
  const absl::string_view reducer_body =
      reducer == ScatterReducerKind::kAdd
          ? "rhs = f32[] parameter(1)\n      ROOT sum = f32[] add(lhs, rhs)"
          : "ROOT rhs = f32[] parameter(1)";
  const std::array<const Literal*, 3> args = {&operand, &indices,
                                              &updates.primary};
  options.alternate_odd_args.clear();
  if (updates.negated.has_value()) {
    options.alternate_odd_args = {nullptr, nullptr, &*updates.negated};
  }
  CHECK_OK(RunHloBenchmark(state, kHlo, args,
                           {{"$reducer_name", reducer_name},
                            {"$reducer_body", reducer_body},
                            {"$num_rows", num_rows_str},
                            {"$slice_size", slice_size_str},
                            {"$num_updates", num_updates_str},
                            {"$unique_indices", unique_str}},
                           options));
  options.alternate_odd_args.clear();
}

template <typename FloatT>
void RunScatterHistLoop53475Acos(benchmark::State& state,
                                 const HloBenchmarkOptions& options,
                                 PrimitiveType type,
                                 absl::string_view hlo_type) {
  constexpr int64_t kTotalElements = 16777216;
  constexpr absl::string_view kHlo = R"(
    HloModule bm_scatter_hist_loop_53475_acos

    add {
      lhs = $dtype[] parameter(0)
      rhs = $dtype[] parameter(1)
      ROOT sum = $dtype[] add(lhs, rhs)
    }

    cond {
      state = (s32[], $dtype[49152], $dtype[16777216]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(8)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    body {
      state = (s32[], $dtype[49152], $dtype[16777216]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      hist = $dtype[49152] get-tuple-element(state), index=1
      x = $dtype[16777216] get-tuple-element(state), index=2

      batch_size = s32[] constant(2097152)
      offset = s32[] multiply(i, batch_size)
      p = $dtype[2097152] dynamic-slice(x, offset),
          dynamic_slice_sizes={2097152}

      two_s = $dtype[] constant(2)
      two = $dtype[2097152] broadcast(two_s), dimensions={}
      one_s = $dtype[] constant(1)
      one = $dtype[2097152] broadcast(one_s), dimensions={}
      scale_s = $dtype[] constant(10000)
      scale = $dtype[2097152] broadcast(scale_s), dimensions={}

      scaled_p = $dtype[2097152] multiply(two, p)
      arg = $dtype[2097152] subtract(scaled_p, one)
      angle = $dtype[2097152] acos(arg)
      scaled_angle = $dtype[2097152] multiply(angle, scale)
      raw = s32[2097152] convert(scaled_angle)

      nbins_s = s32[] constant(49152)
      nbins = s32[2097152] broadcast(nbins_s), dimensions={}
      zero_i_s = s32[] constant(0)
      zero_i = s32[2097152] broadcast(zero_i_s), dimensions={}
      rem = s32[2097152] remainder(raw, nbins)
      is_neg = pred[2097152] compare(rem, zero_i), direction=LT
      wrapped = s32[2097152] add(rem, nbins)
      bins = s32[2097152] select(is_neg, wrapped, rem)
      indices = s32[2097152,1] reshape(bins)

      new_hist = $dtype[49152] scatter(hist, indices, one),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=false,
          to_apply=add

      step = s32[] constant(1)
      next_i = s32[] add(i, step)
      ROOT next_state = (s32[], $dtype[49152], $dtype[16777216])
          tuple(next_i, new_hist, x)
    }

    ENTRY main {
      x = $dtype[16777216] parameter(0)
      init_i = s32[] constant(0)
      zero_s = $dtype[] constant(0)
      init_hist = $dtype[49152] broadcast(zero_s), dimensions={}
      init_state = (s32[], $dtype[49152], $dtype[16777216])
          tuple(init_i, init_hist, x)
      final_state = (s32[], $dtype[49152], $dtype[16777216]) while(init_state),
          condition=cond,
          body=body
      ROOT hist = $dtype[49152] get-tuple-element(final_state), index=1
    }
  )";

  std::minstd_rand0 engine(kBenchmarkSeed);
  const Literal x =
      CreateUniform01Literal<FloatT>(type, kTotalElements, engine);
  const std::vector<const Literal*> args = {&x};
  CHECK_OK(RunHloBenchmark(state, kHlo, args, {{"$dtype", hlo_type}}, options));
}

void BM_ScatterS32_R1(benchmark::State& state, HloBenchmarkOptions options) {
  const int64_t d0 = state.range(0);
  const int64_t slice_size = state.range(1);

  const std::string hlo = R"(
    HloModule BM_ScatterS32_R1

    assign (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1) // just assign the value
    }

    ENTRY main {
      operand = s32[$d0] parameter(0)
      indices = s32[$slice_size] parameter(1)
      updates = s32[$slice_size] parameter(2)
      ROOT scatter = s32[$d0] scatter(operand, indices, updates),
          to_apply=assign,
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }
    )";

  std::minstd_rand0 engine;

  const Shape operand_shape = ShapeUtil::MakeShape(S32, {d0});
  const Literal operand = *LiteralUtil::CreateRandomLiteral<S32>(
      operand_shape, &engine, /*mean=*/50, /*stddev=*/10);

  const Literal scatter_indices =
      createR1ScatterIndices(d0, slice_size, engine);

  const Shape update_shape = ShapeUtil::MakeShape(S32, {slice_size});
  const Literal update = *LiteralUtil::CreateRandomLiteral<S32>(
      update_shape, &engine, /*mean=*/50, /*stddev=*/10);

  std::vector<const Literal*> args = {&operand, &scatter_indices, &update};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$d0", absl::StrCat(d0)}, {"$slice_size", absl::StrCat(slice_size)}},
      options));

  state.SetComplexityN(state.range(1));
}

void BM_ScatterS32_R2(benchmark::State& state, HloBenchmarkOptions options) {
  const int64_t d0 = state.range(0);
  const int64_t d1 = d0;
  const int64_t slice_size = state.range(1);

  const std::string hlo = R"(
    HloModule BM_ScatterS32_R2

    assign (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1) // just assign the value
    }

    ENTRY main {
      operand = s32[$d0,$d1] parameter(0)
      indices = s32[$slice_size] parameter(1)
      updates = s32[$slice_size,$d1] parameter(2)
      ROOT scatter = s32[$d0,$d1] scatter(operand, indices, updates),
          to_apply=assign,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }
    )";

  std::minstd_rand0 engine;

  const Shape operand_shape = ShapeUtil::MakeShape(S32, {d0, d1});
  const Literal operand = *LiteralUtil::CreateRandomLiteral<S32>(
      operand_shape, &engine, /*mean=*/50, /*stddev=*/10);

  const Literal scatter_indices =
      createR1ScatterIndices(d0, slice_size, engine);
  const Shape update_shape = ShapeUtil::MakeShape(S32, {slice_size, d1});
  const Literal update = *LiteralUtil::CreateRandomLiteral<S32>(
      update_shape, &engine, /*mean=*/50, /*stddev=*/10);

  std::vector<const Literal*> args = {&operand, &scatter_indices, &update};
  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {{"$d0", absl::StrCat(d0)},
                            {"$d1", absl::StrCat(d1)},
                            {"$slice_size", absl::StrCat(slice_size)}},
                           options));
}

void BM_ScatterS32_R3(benchmark::State& state, HloBenchmarkOptions options) {
  const int64_t d0 = state.range(0);
  const int64_t d1 = d0;
  const int64_t d2 = d0;
  const int64_t slice_size = state.range(1);

  const std::string hlo = R"(
    HloModule BM_ScatterS32_R3

    assign (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1) // just assign the value
    }

    ENTRY main {
      operand = s32[$d0,$d1,$d2] parameter(0)
      indices = s32[$slice_size] parameter(1)
      updates = s32[$slice_size,$d1,$d2] parameter(2)
      ROOT scatter = s32[$d0,$d1,$d2] scatter(operand, indices, updates),
          to_apply=assign,
          update_window_dims={1, 2},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }
    )";

  std::minstd_rand0 engine;

  const Shape operand_shape = ShapeUtil::MakeShape(S32, {d0, d1, d2});
  const Literal operand = *LiteralUtil::CreateRandomLiteral<S32>(
      operand_shape, &engine, /*mean=*/50, /*stddev=*/10);

  const Literal scatter_indices =
      createR1ScatterIndices(d0, slice_size, engine);

  const Shape update_shape = ShapeUtil::MakeShape(S32, {slice_size, d1, d2});
  const Literal update = *LiteralUtil::CreateRandomLiteral<S32>(
      update_shape, &engine, /*mean=*/50, /*stddev=*/10);

  std::vector<const Literal*> args = {&operand, &scatter_indices, &update};
  CHECK_OK(RunHloBenchmark(state, hlo, args,
                           {{"$d0", absl::StrCat(d0)},
                            {"$d1", absl::StrCat(d1)},
                            {"$d2", absl::StrCat(d2)},
                            {"$slice_size", absl::StrCat(slice_size)}},
                           options));
}

void BM_SimpleScatterReduceF32_R3(benchmark::State& state,
                                  HloBenchmarkOptions options) {
  const int64_t d0 = state.range(0);
  const int64_t d1 = state.range(1);
  const int64_t d2 = state.range(2);
  const int64_t num_reduce_elems = state.range(3);
  const int64_t num_slices = d0 * num_reduce_elems;

  constexpr absl::string_view hlo_string = R"(
    HloModule m
    add {
      %lhs = f32[] parameter(0)
      %rhs = f32[] parameter(1)
      ROOT %add.2 = f32[] add(%lhs, %rhs)
    }
    ENTRY main {
      %operand = f32[$d0,$d1,$d2] parameter(0)
      %indices = s32[$num_slices, 1]{1,0} parameter(1)
      %updates = f32[$num_slices,1,$d1,$d2] parameter(2)
      ROOT %scatter = f32[$d0,$d1,$d2] scatter(%operand, %indices, %updates),
        update_window_dims={1,2,3},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    })";
  std::string hlo = absl::StrReplaceAll(
      hlo_string, {{"$d0", absl::StrCat(d0)},
                   {"$d1", absl::StrCat(d1)},
                   {"$d2", absl::StrCat(d2)},
                   {"$num_slices", absl::StrCat(num_slices)}});

  {
    // Make sure this is a simple scatter.
    absl::StatusOr<std::unique_ptr<HloModule>> module =
        ParseAndReturnUnverifiedModule(hlo);
    CHECK_OK(module.status());
    const HloInstruction* scatter =
        (*module)->entry_computation()->root_instruction();
    CHECK(ScatterSimplifier::IsSimplifiedScatter(
        Cast<HloScatterInstruction>(scatter)));
  }

  std::minstd_rand0 engine;

  const Shape operand_shape = ShapeUtil::MakeShape(F32, {d0, d1, d2});
  const Literal operand = *LiteralUtil::CreateRandomLiteral<F32>(
      operand_shape, &engine, /*mean=*/50, /*stddev=*/10);

  const Literal indices = CreateReduceIndices(d0, num_reduce_elems);

  const Shape update_shape = ShapeUtil::MakeShape(F32, {num_slices, 1, d1, d2});
  const Literal update = *LiteralUtil::CreateRandomLiteral<F32>(
      update_shape, &engine, /*mean=*/50, /*stddev=*/10);

  std::vector<const Literal*> args = {&operand, &indices, &update};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

// Update count scaling benchmarks (unique vs non unique indices, in place add).
void BM_ScatterUniqueAddF32(benchmark::State& state,
                            HloBenchmarkOptions options) {
  const int64_t num_updates = state.range(0);
  const int64_t n_bins = std::max<int64_t>(65536, num_updates);
  RunR1ScatterBenchmark<float>(state, options, F32, "f32", n_bins, num_updates,
                               /*unique_indices=*/true,
                               ScatterReducerKind::kAdd, 1.0f);
}

void BM_ScatterNonUniqueAddF32(benchmark::State& state,
                               HloBenchmarkOptions options) {
  constexpr int64_t kNumBins = 65536;
  const int64_t num_updates = state.range(0);
  RunR1ScatterBenchmark<float>(state, options, F32, "f32", kNumBins,
                               num_updates, /*unique_indices=*/false,
                               ScatterReducerKind::kAdd, 1.0f);
}

// Histogram contention and bin count sweep benchmarks (N = 524288, in place).
void BM_ScatterHistogramAddS32(benchmark::State& state,
                               HloBenchmarkOptions options) {
  constexpr int64_t kNumUpdates = 524288;
  const int64_t n_bins = state.range(0);
  RunR1ScatterBenchmark<int32_t>(state, options, S32, "s32", n_bins,
                                 kNumUpdates, /*unique_indices=*/false,
                                 ScatterReducerKind::kAdd, 1);
}

void BM_ScatterHistogramAddF32(benchmark::State& state,
                               HloBenchmarkOptions options) {
  constexpr int64_t kNumUpdates = 524288;
  const int64_t n_bins = state.range(0);
  RunR1ScatterBenchmark<float>(state, options, F32, "f32", n_bins, kNumUpdates,
                               /*unique_indices=*/false,
                               ScatterReducerKind::kAdd, 1.0f);
}

void BM_ScatterHistogramAddF64(benchmark::State& state,
                               HloBenchmarkOptions options) {
  constexpr int64_t kNumUpdates = 524288;
  const int64_t n_bins = state.range(0);
  RunR1ScatterBenchmark<double>(state, options, F64, "f64", n_bins, kNumUpdates,
                                /*unique_indices=*/false,
                                ScatterReducerKind::kAdd, 1.0);
}

// Rank 0 scalar destination atomic scatter add contention benchmark.
void BM_ScatterRank0AddF64(benchmark::State& state,
                           HloBenchmarkOptions options) {
  const int64_t num_updates = state.range(0);
  constexpr absl::string_view kHlo = R"(
    HloModule bm_scatter_rank0_add_f64, input_output_alias={ {}: (0, {}) }

    add {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT sum = f64[] add(lhs, rhs)
    }

    ENTRY main {
      operand = f64[] parameter(0)
      indices = s32[$num_updates,0] parameter(1)
      updates = f64[$num_updates] parameter(2)
      ROOT scatter = f64[] scatter(operand, indices, updates),
          update_window_dims={},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={},
          index_vector_dim=1,
          unique_indices=false,
          to_apply=add
    }
  )";

  std::minstd_rand0 engine(kBenchmarkSeed);
  const Literal operand = LiteralUtil::CreateR0<double>(0.0);
  const Literal indices =
      LiteralUtil::CreateFromDimensions(S32, {num_updates, 0});
  const std::vector<int32_t> zero_slots(num_updates, 0);
  const BalancedUpdates updates = CreateBalancedUpdates<double>(
      F64, {num_updates}, /*num_slots=*/1, zero_slots, 1.0,
      /*alternate_negated=*/true, engine);

  const std::string num_updates_str = absl::StrCat(num_updates);
  const std::array<const Literal*, 3> args = {&operand, &indices,
                                              &updates.primary};
  if (updates.negated.has_value()) {
    options.alternate_odd_args = {nullptr, nullptr, &*updates.negated};
  }
  CHECK_OK(RunHloBenchmark(state, kHlo, args,
                           {{"$num_updates", num_updates_str}}, options));
}

// Element type and atomic lowering variants (bf16 CAS, c128 CAS, overwrites).
void BM_ScatterHistogramAddBF16(benchmark::State& state,
                                HloBenchmarkOptions options) {
  const int64_t n_bins = state.range(0);
  const int64_t num_updates = state.range(1);
  RunR1ScatterBenchmark<bfloat16>(
      state, options, BF16, "bf16", n_bins, num_updates,
      /*unique_indices=*/false, ScatterReducerKind::kAdd,
      static_cast<bfloat16>(1.0f));
}

void BM_ScatterHistogramAddC128(benchmark::State& state,
                                HloBenchmarkOptions options) {
  const int64_t n_bins = state.range(0);
  const int64_t num_updates = state.range(1);
  RunR1ScatterBenchmark<complex128>(
      state, options, C128, "c128", n_bins, num_updates,
      /*unique_indices=*/false, ScatterReducerKind::kAdd, complex128(1.0, 0.5));
}

void BM_ScatterOverwriteC128(benchmark::State& state,
                             HloBenchmarkOptions options) {
  const int64_t n_bins = state.range(0);
  const int64_t num_updates = state.range(1);
  RunR1ScatterBenchmark<complex128>(
      state, options, C128, "c128", n_bins, num_updates,
      /*unique_indices=*/false, ScatterReducerKind::kOverwrite,
      complex128(1.0, 0.5));
}

void BM_ScatterOverwriteF32(benchmark::State& state,
                            HloBenchmarkOptions options) {
  const int64_t n_bins = state.range(0);
  const int64_t num_updates = state.range(1);
  RunR1ScatterBenchmark<float>(state, options, F32, "f32", n_bins, num_updates,
                               /*unique_indices=*/false,
                               ScatterReducerKind::kOverwrite, 1.0f);
}

void BM_ScatterUniqueOverwriteF32(benchmark::State& state,
                                  HloBenchmarkOptions options) {
  const int64_t num_updates = state.range(0);
  RunR1ScatterBenchmark<float>(state, options, F32, "f32",
                               /*n_bins=*/num_updates, num_updates,
                               /*unique_indices=*/true,
                               ScatterReducerKind::kOverwrite, 1.0f);
}

// Window slice (row scatter and embedding gradient) benchmarks.
void BM_ScatterSliceAddF32(benchmark::State& state,
                           HloBenchmarkOptions options) {
  RunSliceScatterF32Benchmark(state, options, /*unique_indices=*/false,
                              ScatterReducerKind::kAdd);
}

void BM_ScatterSliceUniqueAddF32(benchmark::State& state,
                                 HloBenchmarkOptions options) {
  RunSliceScatterF32Benchmark(state, options, /*unique_indices=*/true,
                              ScatterReducerKind::kAdd);
}

void BM_ScatterSliceOverwriteF32(benchmark::State& state,
                                 HloBenchmarkOptions options) {
  RunSliceScatterF32Benchmark(state, options, /*unique_indices=*/false,
                              ScatterReducerKind::kOverwrite);
}

// Batched histogram while loop benchmarks from openxla/xla#53475.
void BM_ScatterHistLoop53475_F64Acos(benchmark::State& state,
                                     HloBenchmarkOptions options) {
  RunScatterHistLoop53475Acos<double>(state, options, F64, "f64");
}

void BM_ScatterHistLoop53475_F32Acos(benchmark::State& state,
                                     HloBenchmarkOptions options) {
  RunScatterHistLoop53475Acos<float>(state, options, F32, "f32");
}

void BM_ScatterHistLoop53475_F64CheapIndex(benchmark::State& state,
                                           HloBenchmarkOptions options) {
  constexpr int64_t kTotalElements = 16777216;
  constexpr absl::string_view kHlo = R"(
    HloModule bm_scatter_hist_loop_53475_f64_cheap_index

    add {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT sum = f64[] add(lhs, rhs)
    }

    cond {
      state = (s32[], f64[49152], f64[16777216]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(8)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    body {
      state = (s32[], f64[49152], f64[16777216]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      hist = f64[49152] get-tuple-element(state), index=1
      x = f64[16777216] get-tuple-element(state), index=2

      batch_size = s32[] constant(2097152)
      offset = s32[] multiply(i, batch_size)
      p = f64[2097152] dynamic-slice(x, offset), dynamic_slice_sizes={2097152}

      nbins_f_s = f64[] constant(49152)
      nbins_f = f64[2097152] broadcast(nbins_f_s), dimensions={}
      scaled = f64[2097152] multiply(p, nbins_f)
      bins = s32[2097152] convert(scaled)
      indices = s32[2097152,1] reshape(bins)

      one_s = f64[] constant(1)
      ones = f64[2097152] broadcast(one_s), dimensions={}

      new_hist = f64[49152] scatter(hist, indices, ones),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=false,
          to_apply=add

      step = s32[] constant(1)
      next_i = s32[] add(i, step)
      ROOT next_state = (s32[], f64[49152], f64[16777216])
          tuple(next_i, new_hist, x)
    }

    ENTRY main {
      x = f64[16777216] parameter(0)
      init_i = s32[] constant(0)
      zero_s = f64[] constant(0)
      init_hist = f64[49152] broadcast(zero_s), dimensions={}
      init_state = (s32[], f64[49152], f64[16777216])
          tuple(init_i, init_hist, x)
      final_state = (s32[], f64[49152], f64[16777216]) while(init_state),
          condition=cond,
          body=body
      ROOT hist = f64[49152] get-tuple-element(final_state), index=1
    }
  )";

  std::minstd_rand0 engine(kBenchmarkSeed);
  const Literal x = CreateUniform01Literal<double>(F64, kTotalElements, engine);
  const std::vector<const Literal*> args = {&x};
  CHECK_OK(RunHloBenchmark(state, kHlo, args, {}, options));
}

// These all have the same number of elements in the operand:
// (2^18) == (2^9)^2 == (2^6)^3.
XLA_CPU_BENCHMARK(BM_ScatterS32_R1)
    ->MeasureProcessCPUTime()
    ->Args({1 << 18, 1 << 18});
XLA_CPU_BENCHMARK(BM_ScatterS32_R2)
    ->MeasureProcessCPUTime()
    ->Args({1 << 9, 1 << 9});
XLA_CPU_BENCHMARK(BM_ScatterS32_R3)
    ->MeasureProcessCPUTime()
    ->Args({1 << 6, 1 << 6});

XLA_CPU_BENCHMARK(BM_SimpleScatterReduceF32_R3)
    ->MeasureProcessCPUTime()
    ->ArgNames({"d0", "d1", "d2", "num_slices"})
    ->Args({1, 64, 8, 1})
    ->Args({50, 64, 8, 10})
    ->Args({500, 64, 8, 100});

XLA_CPU_BENCHMARK(BM_ScatterUniqueAddF32)
    ->MeasureProcessCPUTime()
    ->ArgName("num_updates")
    ->Arg(256)
    ->Arg(4096)
    ->Arg(32768)
    ->Arg(262144)
    ->Arg(2097152);

XLA_CPU_BENCHMARK(BM_ScatterNonUniqueAddF32)
    ->MeasureProcessCPUTime()
    ->ArgName("num_updates")
    ->Arg(256)
    ->Arg(4096)
    ->Arg(32768)
    ->Arg(262144)
    ->Arg(2097152);

#define REGISTER_HISTOGRAM_BIN_SWEEP(name) \
  XLA_CPU_BENCHMARK(name)                  \
      ->MeasureProcessCPUTime()            \
      ->ArgName("n_bins")                  \
      ->Arg(1)                             \
      ->Arg(8)                             \
      ->Arg(64)                            \
      ->Arg(256)                           \
      ->Arg(1024)                          \
      ->Arg(8192)                          \
      ->Arg(49152)                         \
      ->Arg(1048576)                       \
      ->Arg(8388608)

REGISTER_HISTOGRAM_BIN_SWEEP(BM_ScatterHistogramAddS32);
REGISTER_HISTOGRAM_BIN_SWEEP(BM_ScatterHistogramAddF32);
REGISTER_HISTOGRAM_BIN_SWEEP(BM_ScatterHistogramAddF64);

#undef REGISTER_HISTOGRAM_BIN_SWEEP

XLA_CPU_BENCHMARK(BM_ScatterRank0AddF64)
    ->MeasureProcessCPUTime()
    ->ArgName("num_updates")
    ->Arg(262144);

XLA_CPU_BENCHMARK(BM_ScatterHistogramAddBF16)
    ->MeasureProcessCPUTime()
    ->ArgNames({"n_bins", "num_updates"})
    ->Args({65536, 524288})
    ->Args({16777216, 524288});

XLA_CPU_BENCHMARK(BM_ScatterHistogramAddC128)
    ->MeasureProcessCPUTime()
    ->ArgNames({"n_bins", "num_updates"})
    ->Args({65536, 524288})
    ->Args({2097152, 524288});

XLA_CPU_BENCHMARK(BM_ScatterOverwriteC128)
    ->MeasureProcessCPUTime()
    ->ArgNames({"n_bins", "num_updates"})
    ->Args({65536, 524288});

XLA_CPU_BENCHMARK(BM_ScatterOverwriteF32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"n_bins", "num_updates"})
    ->Args({16, 524288})
    ->Args({65536, 524288})
    ->Args({262144, 524288});

XLA_CPU_BENCHMARK(BM_ScatterUniqueOverwriteF32)
    ->MeasureProcessCPUTime()
    ->ArgName("num_updates")
    ->Arg(262144)
    ->Arg(2097152);

XLA_CPU_BENCHMARK(BM_ScatterSliceAddF32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"num_rows", "slice_size", "num_updates"})
    ->Args({8, 8, 16384})
    ->Args({1024, 64, 16384})
    ->Args({16384, 128, 32768});

XLA_CPU_BENCHMARK(BM_ScatterSliceUniqueAddF32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"num_rows", "slice_size", "num_updates"})
    ->Args({16384, 128, 16384});

XLA_CPU_BENCHMARK(BM_ScatterSliceOverwriteF32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"num_rows", "slice_size", "num_updates"})
    ->Args({16384, 128, 32768});

XLA_CPU_BENCHMARK(BM_ScatterHistLoop53475_F64Acos)->MeasureProcessCPUTime();
XLA_CPU_BENCHMARK(BM_ScatterHistLoop53475_F32Acos)->MeasureProcessCPUTime();
XLA_CPU_BENCHMARK(BM_ScatterHistLoop53475_F64CheapIndex)
    ->MeasureProcessCPUTime();

}  // namespace
}  // namespace xla::cpu
