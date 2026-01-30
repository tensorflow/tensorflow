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

#include "xla/service/cpu/cpu_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/platform.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace cpu {
namespace {

using CpuCompilerTest = HloPjRtTestBase;

constexpr absl::string_view kCpuCompilerStacktraceMetricName =
    "/xla/service/cpu/compiler_stacktrace_count";

TEST_F(CpuCompilerTest, RecordsStreamzStackTrace) {
  if (tsl::kIsOpenSource) {
    GTEST_SKIP() << "Streamz is not supported in OSS.";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
    HloModule test
    ENTRY main {
      p = f32[10]{0} parameter(0)
      ROOT neg = f32[10]{0} negate(p)
    }
  )"));
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));

  tsl::monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<tsl::monitoring::CollectedMetrics> metrics =
      tsl::monitoring::CollectionRegistry::Default()->CollectMetrics(options);

  const auto it = metrics->point_set_map.find(
      std::string(kCpuCompilerStacktraceMetricName));
  ASSERT_TRUE(it != metrics->point_set_map.end());

  // Since Streamz is recorded every call, we expect at least one point.
  // All other callers may increment the counter as well.
  EXPECT_GT(it->second->points.size(), 0);
}

TEST_F(CpuCompilerTest, CompilationWithLargeConstants) {
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[1000,1000]{1,0} parameter(0)
  b = f32[1000,1000]{1,0} constant({...})
  a_plus_b = f32[1000,1000]{1,0} add(a, b)
  c = f32[1000,1000]{1,0} constant({...})
  ROOT result = f32[1000,1000]{1,0} add(a_plus_b, c)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_string));

  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

TEST_F(CpuCompilerTest, IsConcurrencyAffordable) {
  auto is_concurrency_affordable = [](int64_t mem_peak,
                                      int64_t conc_peak) -> bool {
    return CpuCompiler::IsConcurrencyOptimizedScheduleAffordable(mem_peak,
                                                                 conc_peak);
  };
  constexpr int64_t K = 1024;
  constexpr int64_t M = 1024 * K;
  constexpr int64_t G = 1024 * M;

  // Equal memory usage.
  EXPECT_TRUE(is_concurrency_affordable(100 * G, 100 * G));
  // If concurrency-optimized uses less memory, it should always be picked.
  EXPECT_TRUE(is_concurrency_affordable(200 * G, 100 * G));

  // Ratio 1.05, concurrency-optimized-peak 399 G.
  EXPECT_TRUE(is_concurrency_affordable(380 * G, 399 * G));
  EXPECT_FALSE(is_concurrency_affordable(380 * G, 400 * G));

  // Ratio 1.1, concurrency-optimized-peak 99 G.
  EXPECT_TRUE(is_concurrency_affordable(90 * G, 99 * G));
  EXPECT_FALSE(is_concurrency_affordable(90 * G, 100 * G));

  // Ratio 2.0, concurrency-optimized-peak 1 G.
  EXPECT_TRUE(is_concurrency_affordable(512 * M, G));
  EXPECT_FALSE(is_concurrency_affordable(512 * M, G + M));
}

TEST_F(CpuCompilerTest, HonorsSchedulerType) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      p0 = f32[1000,1000]{1,0} parameter(0)
      v1 = f32[1000,1000]{1,0} negate(p0)
      v2 = f32[1000,1000]{1,0} negate(v1)
      v3 = f32[1000,1000]{1,0} negate(p0)
      v4 = f32[1000,1000]{1,0} negate(v3)
      ROOT t = tuple(v2, v4)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CpuCompiler compiler;
  ASSERT_OK_AND_ASSIGN(HloSchedule schedule,
                       compiler.CreateHloSchedule(*module));
  ASSERT_OK(module->set_schedule(schedule));

  auto get_peak_memory =
      [&compiler, &module](DebugOptions::CpuSchedulerType scheduler_type)
      -> absl::StatusOr<int64_t> {
    auto config = module->config();
    auto debug_options = config.debug_options();
    debug_options.set_xla_cpu_scheduler_type(scheduler_type);
    config.set_debug_options(debug_options);
    module->set_config(config);

    ASSIGN_OR_RETURN(auto assignment, compiler.CreateBufferAssignment(*module));
    return assignment->GetStats().total_allocation_bytes;
  };

  ASSERT_OK_AND_ASSIGN(
      int64_t mem_peak,
      get_peak_memory(DebugOptions::CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED));
  ASSERT_OK_AND_ASSIGN(
      int64_t conc_peak,
      get_peak_memory(DebugOptions::CPU_SCHEDULER_TYPE_CONCURRENCY_OPTIMIZED));
  ASSERT_OK_AND_ASSIGN(
      int64_t default_peak,
      get_peak_memory(DebugOptions::CPU_SCHEDULER_TYPE_DEFAULT));

  // The scheduler type option should affect the buffer assignment.
  EXPECT_NE(mem_peak, conc_peak);
  // For this small graph, concurrency optimized should be affordable and thus
  // selected by the default policy.
  EXPECT_EQ(default_peak, conc_peak);
}

TEST_F(CpuCompilerTest, CollectivesForceConcurrencyOptimized) {
  // We construct a graph with 5 parallel chains where the concurrency-optimized
  // schedule uses significantly more memory than the memory-optimized schedule.
  //
  // Graph structure per chain i:
  //   t1_i = negate(p)
  //   t2_i = negate(t1_i)
  //   t3_i = add(t1_i, t2_i) <-- Forces t1_i and t2_i to be live
  //   simultaneously.
  //
  // Buffer size: 400MB (10000x10000 f32).
  //
  // Sequential Schedule (Chain 1 -> Chain 2 ...):
  //   Max Overlap: t1_i, t2_i, results_so_far.
  //   Intermediates: 2 buffers (reused across chains).
  //
  // Concurrency Schedule (Dependency Ordering):
  //   All t1_i and t2_i are unordered with respect to each other.
  //   Intermediates: 5 * 2 = 10 buffers.
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    ENTRY main {
      p = f32[10000,10000] parameter(0)

      // Chain 1
      c1_1 = f32[10000,10000] negate(p)
      c1_2 = f32[10000,10000] negate(c1_1)
      c1_3 = f32[10000,10000] add(c1_1, c1_2)

      // Chain 2
      c2_1 = f32[10000,10000] negate(p)
      c2_2 = f32[10000,10000] negate(c2_1)
      c2_3 = f32[10000,10000] add(c2_1, c2_2)

      // Chain 3
      c3_1 = f32[10000,10000] negate(p)
      c3_2 = f32[10000,10000] negate(c3_1)
      c3_3 = f32[10000,10000] add(c3_1, c3_2)

      // Chain 4
      c4_1 = f32[10000,10000] negate(p)
      c4_2 = f32[10000,10000] negate(c4_1)
      c4_3 = f32[10000,10000] add(c4_1, c4_2)

      // Chain 5
      c5_1 = f32[10000,10000] negate(p)
      c5_2 = f32[10000,10000] negate(c5_1)
      c5_3 = f32[10000,10000] add(c5_1, c5_2)

      // Collective
      zero = f32[] constant(0)
      cr = f32[] all-reduce(zero), replica_groups={{0}}, to_apply=add

      ROOT t = tuple(c1_3, c2_3, c3_3, c4_3, c5_3, cr)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CpuCompiler compiler;
  // Manually enforce a sequential schedule.
  auto* entry = module->entry_computation();
  std::vector<HloInstruction*> seq_order;
  seq_order.push_back(entry->parameter_instruction(0));
  auto add_chain = [&](int chain_idx) {
    std::string prefix = "c" + std::to_string(chain_idx) + "_";
    seq_order.push_back(entry->GetInstructionWithName(prefix + "1"));
    seq_order.push_back(entry->GetInstructionWithName(prefix + "2"));
    seq_order.push_back(entry->GetInstructionWithName(prefix + "3"));
  };
  add_chain(1);
  add_chain(2);
  add_chain(3);
  add_chain(4);
  add_chain(5);
  seq_order.push_back(entry->GetInstructionWithName("zero"));
  seq_order.push_back(entry->GetInstructionWithName("cr"));
  seq_order.push_back(entry->root_instruction());
  ASSERT_OK_AND_ASSIGN(HloSchedule schedule,
                       compiler.CreateHloSchedule(*module));
  schedule.set_sequence(entry, seq_order);
  ASSERT_OK(module->set_schedule(schedule));

  auto get_peak_memory =
      [&compiler, &module](DebugOptions::CpuSchedulerType scheduler_type)
      -> absl::StatusOr<int64_t> {
    auto config = module->config();
    auto debug_options = config.debug_options();
    debug_options.set_xla_cpu_scheduler_type(scheduler_type);
    config.set_debug_options(debug_options);
    module->set_config(config);

    ASSIGN_OR_RETURN(auto assignment, compiler.CreateBufferAssignment(*module));
    return assignment->GetStats().total_allocation_bytes;
  };

  ASSERT_OK_AND_ASSIGN(
      int64_t mem_peak,
      get_peak_memory(DebugOptions::CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED));
  ASSERT_OK_AND_ASSIGN(
      int64_t conc_peak,
      get_peak_memory(DebugOptions::CPU_SCHEDULER_TYPE_CONCURRENCY_OPTIMIZED));
  ASSERT_OK_AND_ASSIGN(
      int64_t default_peak,
      get_peak_memory(DebugOptions::CPU_SCHEDULER_TYPE_DEFAULT));

  // Concurrency optimized takes more memory.
  EXPECT_GT(conc_peak, mem_peak);
  // Concurrency is not affordable.
  EXPECT_FALSE(CpuCompiler::IsConcurrencyOptimizedScheduleAffordable(
      mem_peak, conc_peak));
  // The default matches the concurrency-optimized schedule because the module
  // has a collective.
  EXPECT_EQ(default_peak, conc_peak);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
