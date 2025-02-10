/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/schedule_postprocessing.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using SchedulePostprocessingTest = HloTestBase;

TEST_F(SchedulePostprocessingTest, SynchronousOpsNotChanged) {
  constexpr absl::string_view kHloString = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    pf32 = f32[1] parameter(0)

    all-gather-start = (f32[1], f32[2]) all-gather-start(pf32), dimensions={0}, backend_config={"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false}}
    ROOT all-gather-done = f32[2] all-gather-done(all-gather-start)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kHloString)));
  SchedulePostprocessing pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SchedulePostprocessingTest, P2POpsNotChanged) {
  constexpr absl::string_view kHloString = R"(
  HloModule module, is_scheduled=true

  ENTRY main {
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=2,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0,1}, {1,2}}"
    }
    recv-done = (f32[1, 1024, 1024], token[]) recv-done(recv), channel_id=2
    ROOT recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done), index=0
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kHloString)));
  SchedulePostprocessing pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SchedulePostprocessingTest, AsynchronousOpsChanged) {
  constexpr absl::string_view kHloString = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    pf32 = f32[1] parameter(0)
    pf32.2 = f32[1] custom-call(pf32), custom_call_target="my_custom_call"
    all-gather-start = (f32[1], f32[2]) all-gather-start(pf32.2), dimensions={0}, backend_config={"collective_backend_config":{"is_sync":false}}
    ROOT all-gather-done = f32[2] all-gather-done(all-gather-start)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kHloString)));
  SchedulePostprocessing pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* start = FindInstruction(module.get(), "all-gather-start");
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          start->backend_config<GpuBackendConfig>());
  const CollectiveBackendConfig& collective_backend_config =
      gpu_config.collective_backend_config();
  EXPECT_TRUE(collective_backend_config.no_parallel_custom_call());
}

TEST_F(SchedulePostprocessingTest, AsynchronousOpsWithParallelCustomcall) {
  constexpr absl::string_view kHloString = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    pf32 = f32[1] parameter(0)
    all-gather-start = (f32[1], f32[2]) all-gather-start(pf32), dimensions={0}, backend_config={"collective_backend_config":{"is_sync":false}}
    pf32.2 = f32[1] custom-call(pf32), custom_call_target="my_custom_call"
    all-gather-done = f32[2] all-gather-done(all-gather-start)
    ROOT out = (f32[1], f32[2]) tuple(f32[1] pf32.2, f32[2] all-gather-done)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kHloString)));
  SchedulePostprocessing pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);

  HloInstruction* start = FindInstruction(module.get(), "all-gather-start");
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          start->backend_config<GpuBackendConfig>());
  const CollectiveBackendConfig& collective_backend_config =
      gpu_config.collective_backend_config();
  EXPECT_FALSE(collective_backend_config.no_parallel_custom_call());
}

TEST_F(SchedulePostprocessingTest,
       AsynchronousOpsWithParallelNestedCustomcall) {
  constexpr absl::string_view kHloString = R"(
  HloModule module, is_scheduled=true
  foo {
    v = f32[1] parameter(0)
    ROOT ret = f32[1] custom-call(v), custom_call_target="my_custom_call"
  }

  ENTRY entry {
    pf32 = f32[1] parameter(0)
    all-gather-start = (f32[1], f32[2]) all-gather-start(pf32), dimensions={0}, backend_config={"collective_backend_config":{"is_sync":false}}
    pf32.2 = f32[1] call(f32[1] pf32), to_apply=foo
    all-gather-done = f32[2] all-gather-done(all-gather-start)
    ROOT out = (f32[1], f32[2]) tuple(f32[1] pf32.2, f32[2] all-gather-done)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kHloString)));
  SchedulePostprocessing pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);

  HloInstruction* start = FindInstruction(module.get(), "all-gather-start");
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          start->backend_config<GpuBackendConfig>());
  const CollectiveBackendConfig& collective_backend_config =
      gpu_config.collective_backend_config();
  EXPECT_FALSE(collective_backend_config.no_parallel_custom_call());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
