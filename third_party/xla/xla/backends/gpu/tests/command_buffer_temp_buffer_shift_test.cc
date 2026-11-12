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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// Long chain of GEMMs with cosine and add operations to ensure a large number
// of temporary buffers.
std::string BuildChainedGemmHlo(int64_t num_replicas) {
  constexpr int kNumRepeats = 500;
  std::string hlo = absl::StrFormat(R"(
HloModule temp_buffer_shift_repro, replica_count=%d, input_output_alias={ {}: (0, {}, must-alias) }
ENTRY main {
  p0 = f32[64,64] parameter(0)
)",
                                    num_replicas);
  std::string cur = "p0";
  for (int i = 0; i < kNumRepeats; ++i) {
    absl::StrAppendFormat(
        &hlo,
        "  dot_%d = f32[64,64] dot(%s, p0), lhs_contracting_dims={1}, "
        "rhs_contracting_dims={0}\n"
        "  cos_%d = f32[64,64] cosine(dot_%d)\n"
        "  act_%d = f32[64,64] add(%s, cos_%d)\n",
        i, cur, i, i, i, cur, i);
    cur = absl::StrFormat("act_%d", i);
  }
  absl::StrAppendFormat(&hlo, "  ROOT out = f32[64,64] cosine(%s)\n}\n", cur);
  return hlo;
}

struct LaunchStats {
  double mean_us = 0.0;
  double stddev_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
};

LaunchStats ComputeLaunchStats(const std::vector<double>& samples) {
  double sum = 0.0;
  double min_val = samples[0];
  double max_val = samples[0];
  for (double x : samples) {
    sum += x;
    min_val = std::min(min_val, x);
    max_val = std::max(max_val, x);
  }
  const double mean = sum / samples.size();
  double sq_diff_sum = 0.0;
  for (double x : samples) {
    const double diff = x - mean;
    sq_diff_sum += diff * diff;
  }
  const double stddev = std::sqrt(sq_diff_sum / samples.size());
  return {mean, stddev, min_val, max_val};
}

class CommandBufferTempBufferShiftTest : public CollectiveOpsE2ETestBase {
 public:
  CommandBufferTempBufferShiftTest()
      : CollectiveOpsE2ETestBase(/*memory_size=*/2 * kGB,
                                 /*collectives_memory_size=*/0) {}

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        CollectiveOpsE2ETestBase::GetDebugOptionsForTest();
    // Updating cublas graph nodes is more expensive, so we disable Triton GEMM.
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    debug_options.set_xla_gpu_temp_buffer_use_collective_color(true);
    return debug_options;
  }
};

TEST_F(CommandBufferTempBufferShiftTest,
       TempBufferShiftOnBfcAllocatorCausesGraphUpdateOverheadOnAllGpus) {
  // Temp buffers only go to the collective memory space when the module is
  // fully sharded, i.e. it spans every visible device. Use all of them so the
  // test behaves the same on 2-GPU and 8-GPU hosts.
  const int64_t num_replicas = device_count();
  if (num_replicas < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/num_replicas, /*num_partitions=*/1);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(BuildChainedGemmHlo(num_replicas), config));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));

  auto* runner = static_cast<HloRunner*>(&test_runner());
  Literal p0 =
      LiteralUtil::CreateFullWithDescendingLayout<float>({64, 64}, 0.01f);
  Shape param_shape = ShapeUtil::MakeShape(F32, {64, 64});
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<PjRtBuffer>> param_buffers,
                       runner->TransferLiteralsToDefaultDevice(
                           {ShapeLayout(param_shape)}, {&p0}));
  PjRtClient* pjrt_client = param_buffers[0]->client();
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * mem_space,
      pjrt_client->addressable_devices()[0]->default_memory_space());

  auto run_launch = [&]() -> absl::StatusOr<double> {
    const absl::Time t0 = absl::Now();
    ABSL_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<PjRtBuffer>> results,
        runner->ExecuteWithDeviceBuffers(executable.get(), param_buffers));
    const double elapsed_us = absl::ToDoubleMicroseconds(absl::Now() - t0);
    // Parameter 0 is aliased (must-alias) to the root output and donated in
    // place, so results[0] wraps the same underlying device allocation.
    param_buffers[0] = std::move(results[0]);
    return elapsed_us;
  };

  // 1. Warmup: initialize and record the large CUDA Graph.
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK_AND_ASSIGN(double warmup_us, run_launch());
    VLOG(1) << absl::StrFormat("Warmup %d: %.1f us", i, warmup_us);
  }

  // 2. Run 10 baseline launches with a stable temp buffer.
  constexpr int kNumLaunches = 10;
  std::vector<double> baseline_samples_us;
  baseline_samples_us.reserve(kNumLaunches);
  for (int i = 0; i < kNumLaunches; ++i) {
    ASSERT_OK_AND_ASSIGN(double launch_us, run_launch());
    baseline_samples_us.push_back(launch_us);
    VLOG(1) << absl::StrFormat("Baseline %d: %.1f us", i, launch_us);
  }
  const LaunchStats baseline = ComputeLaunchStats(baseline_samples_us);

  // 3. Run 10 launches where before each launch we allocate a persistent buffer
  // on BFCAllocator so that if the preallocated temp buffer (`alloc[1]`)
  // resides on BFCAllocator's default memory space, its address shifts on every
  // launch while `p0` (`alloc[0]`) stays pinned at its baseline address.
  Shape temp_shift_shape = ShapeUtil::MakeShape(U8, {34 * 1024 * 1024});
  std::vector<std::unique_ptr<PjRtBuffer>> held_bfc_buffers;
  held_bfc_buffers.reserve(kNumLaunches);
  std::vector<double> post_shift_samples_us;
  post_shift_samples_us.reserve(kNumLaunches);
  for (int i = 0; i < kNumLaunches; ++i) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtBuffer> shift_buf,
        pjrt_client->CreateUninitializedBuffer(temp_shift_shape, mem_space));
    held_bfc_buffers.push_back(std::move(shift_buf));
    ASSERT_OK_AND_ASSIGN(double post_shift_us, run_launch());
    post_shift_samples_us.push_back(post_shift_us);
    VLOG(1) << absl::StrFormat("Post-shift %d: %.1f us", i, post_shift_us);
  }
  const LaunchStats post_shift = ComputeLaunchStats(post_shift_samples_us);
  const double delta_us = post_shift.mean_us - baseline.mean_us;

  VLOG(1) << absl::StrFormat(
      "Baseline (%d launches)  : mean = %.1f us (%.3f ms), "
      "stddev = %.1f us (%.1f%%), min = %.1f us, max = %.1f us\n"
      "Post-shift (%d launches): mean = %.1f us (%.3f ms), "
      "stddev = %.1f us (%.1f%%), min = %.1f us, max = %.1f us\n"
      "Additional overhead     : +%.1f us (+%.3f ms, %.2fx baseline)",
      kNumLaunches, baseline.mean_us, baseline.mean_us / 1000.0,
      baseline.stddev_us, (baseline.stddev_us / baseline.mean_us) * 100.0,
      baseline.min_us, baseline.max_us, kNumLaunches, post_shift.mean_us,
      post_shift.mean_us / 1000.0, post_shift.stddev_us,
      (post_shift.stddev_us / post_shift.mean_us) * 100.0, post_shift.min_us,
      post_shift.max_us, delta_us, delta_us / 1000.0,
      post_shift.mean_us / baseline.mean_us);

  EXPECT_LT(post_shift.mean_us, baseline.mean_us * 1.10);
}

}  // namespace
}  // namespace xla
