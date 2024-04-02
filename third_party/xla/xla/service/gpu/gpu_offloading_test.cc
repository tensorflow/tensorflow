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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotune_results.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/layout.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/hlo_rematerialization.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;
namespace op = xla::testing::opcode_matchers;

using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::TempDir;

class GpuCompilerTest : public HloTestBase {
 protected:
  absl::StatusOr<bool> RunHloRematerialization(int64_t memory_limit_bytes,
                                               HloModule* module,
                                               int64_t min_remat_size = 0) {
    TF_EXPECT_OK(verifier().Run(module).status());
    if (!module->has_schedule()) {
      HloMemoryScheduler scheduler(
          [](const BufferValue& buffer) {
            return ::xla::ShapeUtil::ByteSizeOf(buffer.shape());
          },
          ComputationSchedulerToModuleScheduler(DefaultMemoryScheduler));
      TF_EXPECT_OK(scheduler.Run(module).status());
    }
    // Create a configuration where any compute is much much slower than any
    // number of number of copies.
    HloCostAnalysis::Options hlo_cost_analysis_options;
    hlo_cost_analysis_options.shape_size = [](const Shape& shape) {
      return ::xla::ShapeUtil::ByteSizeOf(shape);
    };
    hlo_cost_analysis_options.set_flops_per_second(flops_per_second_);
    hlo_cost_analysis_options.set_transcendentals_per_second(
        transcendentals_per_second_);
    HloCostAnalysis cost_analysis(hlo_cost_analysis_options);
    HloRematerialization::RematerializationModeConfig config(
        /*recompute=*/false, /*compress=*/false, /*host_offload=*/true);
    HloRematerialization::HostMemoryOffloadConfig host_memory_offload_config(
        kHostMemorySpaceColor, copy_to_host_speed_, copy_from_host_speed_);
    HloRematerialization::Options options(
        cost_analysis, config, memory_limit_bytes,
        /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
        min_remat_size, /*compact_shape_function=*/nullptr,
        host_memory_offload_config);
    HloRematerialization::RematerializationSizes sizes;
    HloRematerialization remat(options, sizes);
    return remat.Run(module);
  }
  void SetCopyToHostSpeed(float val) { copy_to_host_speed_ = val; }
  void SetCopyFromHostSpeed(float val) { copy_from_host_speed_ = val; }
  void SetFlopsPerSecond(float val) { flops_per_second_ = val; }
  void SetTranscendentalsPerSecond(float val) {
    transcendentals_per_second_ = val;
  }

  static constexpr const int64_t kHostMemorySpaceColor{5};

 private:
  float copy_to_host_speed_{1.0f};
  float copy_from_host_speed_{1.0f};
  float flops_per_second_{1.0f};
  float transcendentals_per_second_{1.0f};
};

TEST_F(GpuCompilerTest, OriginalTest) {
  const char* hlo_text = R"(
  HloModule test

ENTRY %main (param_0: f32[1024], param_1: f32[1024]) -> f32[1024] {
  %param_1 = f32[1024]{0} parameter(1)
  %param_0 = f32[1024]{0} parameter(0)
  %res_3 = f32[1024]{0} add(f32[1024]{0} %param_0, f32[1024]{0} %param_1)
  %copy-start = (f32[1024]{0:S(5)}, f32[1024]{0}, u32[]) copy-start(f32[1024]{0} %res_3)
  %res_4 = f32[1024]{0} tanh(f32[1024]{0} %res_3)
  %copy-start.2 = (f32[1024]{0:S(5)}, f32[1024]{0}, u32[]) copy-start(f32[1024]{0} %res_4)
  %res_5 = f32[1024]{0} tanh(f32[1024]{0} %res_4)
  %copy-done = f32[1024]{0:S(5)} copy-done((f32[1024]{0:S(5)}, f32[1024]{0}, u32[]) %copy-start)
  %res_6 = f32[1024]{0} tanh(f32[1024]{0} %res_5)
  %copy-done.2 = f32[1024]{0:S(5)} copy-done((f32[1024]{0:S(5)}, f32[1024]{0}, u32[]) %copy-start.2)
  %copy-start.3 = (f32[1024]{0}, f32[1024]{0:S(5)}, u32[]) copy-start(f32[1024]{0:S(5)} %copy-done.2)
  %res_7 = f32[1024]{0} add(f32[1024]{0} %res_6, f32[1024]{0} %res_6)
  %copy-start.1 = (f32[1024]{0}, f32[1024]{0:S(5)}, u32[]) copy-start(f32[1024]{0:S(5)} %copy-done)
  %res_8 = f32[1024]{0} add(f32[1024]{0} %res_7, f32[1024]{0} %res_5)
  %copy-done.3 = f32[1024]{0} copy-done((f32[1024]{0}, f32[1024]{0:S(5)}, u32[]) %copy-start.3)
  %res_9 = f32[1024]{0} add(f32[1024]{0} %res_8, f32[1024]{0} %copy-done.3)
  %copy-done.1 = f32[1024]{0} copy-done((f32[1024]{0}, f32[1024]{0:S(5)}, u32[]) %copy-start.1)
  %res_10 = f32[1024]{0} add(f32[1024]{0} %res_9, f32[1024]{0} %copy-done.1)
  ROOT %res_11 = f32[1024]{0} tanh(f32[1024]{0} %res_10)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(GpuCompilerTest, CompiledProgramsCount) {
  const char* hlo_text = R"(
  HloModule test

  ENTRY main {
    param_0 = f32[1024]{0} parameter(0)
    param_1 = f32[1024]{0} parameter(1)
    res_3 = f32[1024]{0} add(param_0, param_1)
    res_4 = f32[1024]{0} tanh(res_3)
    res_5 = f32[1024]{0} tanh(res_4)
    res_6 = f32[1024]{0} tanh(res_5)
    res_7 = f32[1024]{0} add(res_6, res_6)
    res_8 = f32[1024]{0} add(res_7, res_5)
    res_9 = f32[1024]{0} add(res_8, res_4)
    res_10 = f32[1024]{0} add(res_9, res_3)
    ROOT res_11 = f32[1024]{0} tanh(res_10)
  }
)";

  auto module = ParseAndReturnVerifiedModule(hlo_text).value();
  auto module_ref = ParseAndReturnVerifiedModule(hlo_text).value();

  // Set some "hardware" constants so that we can test that instructions are
  // placed in the places we expect.
  SetCopyToHostSpeed(4.0 * 1024);
  SetCopyFromHostSpeed(4.0 * 1024);
  SetFlopsPerSecond(2 * 1024);
  SetTranscendentalsPerSecond(2 * 1024);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/10 * 1024, module.get()));
  ASSERT_TRUE(changed);

  // The module should still have a schedule.
  ASSERT_TRUE(module->has_schedule());

  // Verify that exactly two instructions are rematerialized.
  auto res_3_matcher = op::Add(op::Parameter(), op::Parameter());
  auto res_3_rematted_matcher = op::AsyncCopy(
      xla::Layout::kDefaultMemorySpace, kHostMemorySpaceColor,
      op::AsyncCopy(kHostMemorySpaceColor, xla::Layout::kDefaultMemorySpace,
                    res_3_matcher));
  auto res_4_matcher = op::Tanh(res_3_matcher);
  auto res_4_rematted_matcher = op::AsyncCopy(
      xla::Layout::kDefaultMemorySpace, kHostMemorySpaceColor,
      op::AsyncCopy(kHostMemorySpaceColor, xla::Layout::kDefaultMemorySpace,
                    res_4_matcher));
  auto res_5_matcher = op::Tanh(res_4_matcher);
  auto res_6_matcher = op::Tanh(res_5_matcher);
  auto res_7_matcher = op::Add(res_6_matcher, res_6_matcher);
  auto res_8_matcher = op::Add(res_7_matcher, res_5_matcher);
  auto res_9_matcher = op::Add(res_8_matcher, res_4_rematted_matcher);
  auto res_10_matcher = op::Add(res_9_matcher, res_3_rematted_matcher);

  const auto instruction_sequence =
      module->schedule().sequence(module->entry_computation());
  ASSERT_THAT(instruction_sequence.instructions().back(),
              op::Tanh(res_10_matcher));
  // module has the graph optimized by rematerialization and schedule
  // module_ref has the original graph without rematerialization
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module_ref),
                                      ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6},
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
