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

#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

using ::testing::HasSubstr;

struct RecordedSnapshots {
  absl::Mutex mu;
  std::vector<std::array<float, 4>> values ABSL_GUARDED_BY(mu);
};

RecordedSnapshots& GetRecordedSnapshots() {
  static absl::NoDestructor<RecordedSnapshots> snapshots;
  return *snapshots;
}

absl::Status RecordBf16Bins(
    ffi::BufferR1<PrimitiveType::BF16> in,
    ffi::Result<ffi::BufferR1<PrimitiveType::BF16>> out) {
  std::array<float, 4> snap = {};
  for (int i = 0; i < 4; ++i) {
    out->typed_data()[i] = in.typed_data()[i];
    snap[i] = static_cast<float>(in.typed_data()[i]);
  }
  RecordedSnapshots& rec = GetRecordedSnapshots();
  absl::MutexLock lock(rec.mu);
  rec.values.push_back(snap);
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kRecordBf16Bins, RecordBf16Bins,
                       ffi::Ffi::Bind()
                           .Arg<ffi::BufferR1<PrimitiveType::BF16>>()
                           .Ret<ffi::BufferR1<PrimitiveType::BF16>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_bm_test$$record_bf16_bins",
                         "Host", kRecordBf16Bins);

constexpr absl::string_view kAliasedBf16ScatterHlo = R"hlo(
  HloModule aliased_bf16_scatter, input_output_alias={ {}: (0, {}) }

  add {
    lhs = bf16[] parameter(0)
    rhs = bf16[] parameter(1)
    ROOT sum = bf16[] add(lhs, rhs)
  }

  ENTRY main {
    operand = bf16[4] parameter(0)
    indices = s32[4,1] parameter(1)
    updates = bf16[4] parameter(2)
    scattered = bf16[4] scatter(operand, indices, updates),
        update_window_dims={}, inserted_window_dims={0},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=add
    ROOT observed = bf16[4] custom-call(scattered),
        custom_call_target="__xla_bm_test$$record_bf16_bins",
        api_version=API_VERSION_TYPED_FFI,
        output_to_operand_aliasing={{}: (0, {})}
  }
)hlo";

TEST(HloBenchmarkRunnerTest, AlternateOddArgsValidation) {
  const Literal operand = LiteralUtil::CreateFromDimensions(BF16, {4});
  const Literal indices = LiteralUtil::CreateR2<int32_t>({{0}, {1}, {1}, {2}});
  const Literal updates = LiteralUtil::CreateR1<bfloat16>(
      {bfloat16(1.0f), bfloat16(1.0f), bfloat16(-1.0f), bfloat16(-1.0f)});
  const Literal wrong_shape = LiteralUtil::CreateFromDimensions(BF16, {2});
  const std::array<const Literal*, 3> args = {&operand, &indices, &updates};

  auto run_once = [&](absl::Span<const Literal* const> run_args,
                      std::vector<const Literal*> alt_args) {
    absl::StatusOr<std::unique_ptr<HloModule>> module =
        ParseAndReturnUnverifiedModule(kAliasedBf16ScatterHlo);
    if (!module.ok()) {
      return module.status();
    }
    HloBenchmarkOptions opts;
    opts.alternate_odd_args = std::move(alt_args);
    return RunHloBenchmarkOnce(*std::move(module), run_args, opts);
  };

  EXPECT_THAT(run_once({}, {nullptr, nullptr, &updates}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("requires explicit args")));

  EXPECT_THAT(run_once(args, {nullptr}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("alternate_odd_args")));

  EXPECT_THAT(run_once(args, {&operand, nullptr, nullptr}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("aliased parameter")));

  EXPECT_THAT(run_once(args, {nullptr, nullptr, &wrong_shape}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("does not match primary")));
}

TEST(HloBenchmarkRunnerTest, AlternateOddArgsCancelsAliasedBf16Accumulation) {
  {
    RecordedSnapshots& rec = GetRecordedSnapshots();
    absl::MutexLock lock(rec.mu);
    rec.values.clear();
  }

  // Run past bf16's +-256 unit step saturation point (1 warmup + 600 timed).
  constexpr int64_t kTimedIterations = 600;
  int argc = 1;
  char arg0[] = "hlo_benchmark_runner_test";
  char* argv[] = {arg0, nullptr};
  benchmark::Initialize(&argc, argv);

  absl::Status run_status;
  benchmark::RegisterBenchmark(
      "BM_TestAlternateOddArgsBf16",
      [&run_status](benchmark::State& state) {
        const Literal operand = LiteralUtil::CreateFromDimensions(BF16, {4});
        const Literal indices =
            LiteralUtil::CreateR2<int32_t>({{0}, {1}, {1}, {2}});
        const Literal pos_updates = LiteralUtil::CreateR1<bfloat16>(
            {bfloat16(1.0f), bfloat16(1.0f), bfloat16(-1.0f), bfloat16(-1.0f)});
        const Literal neg_updates = LiteralUtil::CreateR1<bfloat16>(
            {bfloat16(-1.0f), bfloat16(-1.0f), bfloat16(1.0f), bfloat16(1.0f)});
        const std::array<const Literal*, 3> args = {&operand, &indices,
                                                    &pos_updates};
        HloBenchmarkOptions opts;
        opts.alternate_odd_args = {nullptr, nullptr, &neg_updates};
        run_status =
            RunHloBenchmark(state, kAliasedBf16ScatterHlo, args, {}, opts);
      })
      ->Iterations(kTimedIterations);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::ClearRegisteredBenchmarks();
  ASSERT_OK(run_status);

  RecordedSnapshots& rec = GetRecordedSnapshots();
  absl::MutexLock lock(rec.mu);
  ASSERT_EQ(rec.values.size(), kTimedIterations + 1);
  constexpr std::array<float, 4> kAfterEven = {1.0f, 0.0f, -1.0f, 0.0f};
  constexpr std::array<float, 4> kAfterOdd = {0.0f, 0.0f, 0.0f, 0.0f};
  for (size_t i = 0; i < rec.values.size(); ++i) {
    EXPECT_EQ(rec.values[i], (i % 2 == 0) ? kAfterEven : kAfterOdd) << i;
  }
}

}  // namespace
}  // namespace xla::cpu
