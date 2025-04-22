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

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::IsFalse;
using ::tsl::testing::IsOk;

using BackendConfigsTest = HloHardwareIndependentTestBase;

TEST_F(BackendConfigsTest, DefaultCollectiveBackendConfig) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    pf32 = f32[1] parameter(0)

    agf32-start = (f32[1], f32[2]) all-gather-start(pf32), dimensions={0}
    ROOT agf32-done = f32[2] all-gather-done(agf32-start)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloString, /*replica_count=*/2));

  const HloInstruction* ags = FindInstruction(module.get(), "agf32-start");
  EXPECT_THAT(ags->has_backend_config(), IsFalse());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          ags->backend_config<GpuBackendConfig>());
  const auto& collective_backend_config =
      gpu_config.collective_backend_config();
  EXPECT_THAT(collective_backend_config.is_sync(), IsFalse());
}

TEST_F(BackendConfigsTest, DefaultGpuBackendConfigParseOpQueue) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p0f32 = f32[4, 4] parameter(0)
    p1f32 = f32[4, 4] parameter(1)

    ROOT addf32 = f32[4, 4] add(p0f32, p1f32), backend_config={"operation_queue_id":"2"}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  HloInstruction* add = module->entry_computation()->root_instruction();
  EXPECT_TRUE(add->has_backend_config());
  auto real_gpu_backend_config = add->backend_config<GpuBackendConfig>();
  EXPECT_THAT(real_gpu_backend_config.status(), IsOk());
  EXPECT_EQ(real_gpu_backend_config->operation_queue_id(), 2);
}

TEST_F(BackendConfigsTest, DefaultGpuBackendConfigParseWaitOnQueue) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p0f32 = f32[4, 4] parameter(0)
    p1f32 = f32[4, 4] parameter(1)

    ROOT addf32 = f32[4, 4] add(p0f32, p1f32), backend_config={"wait_on_operation_queues":[0, 1]}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  HloInstruction* add = module->entry_computation()->root_instruction();
  EXPECT_TRUE(add->has_backend_config());
  auto real_gpu_backend_config = add->backend_config<GpuBackendConfig>();
  EXPECT_THAT(real_gpu_backend_config.status(), IsOk());
  std::vector<int64_t> expected_ids = {0, 1};
  EXPECT_EQ(real_gpu_backend_config->wait_on_operation_queues().size(),
            expected_ids.size());
  for (int64_t i = 0; i < expected_ids.size(); i++) {
    EXPECT_EQ(expected_ids[i],
              real_gpu_backend_config->wait_on_operation_queues()[i]);
  }
}

TEST_F(BackendConfigsTest, DefaultGpuBackendConfigSetOpQueue) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p0f32 = f32[4, 4] parameter(0)
    p1f32 = f32[4, 4] parameter(1)

    ROOT addf32 = f32[4, 4] add(p0f32, p1f32)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  HloInstruction* add = module->entry_computation()->root_instruction();
  EXPECT_FALSE(add->has_backend_config());
  GpuBackendConfig gpu_backend_config;
  gpu_backend_config.set_operation_queue_id(2);
  EXPECT_THAT(add->set_backend_config(gpu_backend_config), IsOk());
  EXPECT_EQ(add->raw_backend_config_string(),
            "{\"operation_queue_id\":\"2\",\"wait_on_operation_queues\":[],"
            "\"force_earliest_schedule\":false}");
}

TEST_F(BackendConfigsTest, DefaultGpuBackendConfigSetWaitOnQueue) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p0f32 = f32[4, 4] parameter(0)
    p1f32 = f32[4, 4] parameter(1)

    ROOT addf32 = f32[4, 4] add(p0f32, p1f32)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  HloInstruction* add = module->entry_computation()->root_instruction();
  EXPECT_FALSE(add->has_backend_config());
  GpuBackendConfig gpu_backend_config;
  // Wait on queues {0, 1}
  gpu_backend_config.mutable_wait_on_operation_queues()->Add(0);
  gpu_backend_config.mutable_wait_on_operation_queues()->Add(1);
  EXPECT_THAT(add->set_backend_config(gpu_backend_config), IsOk());
  EXPECT_EQ(add->raw_backend_config_string(),
            "{\"operation_queue_id\":\"0\",\"wait_on_operation_queues\":[\"0\","
            "\"1\"],\"force_earliest_schedule\":false}");
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig config,
                          add->backend_config<GpuBackendConfig>());
}

TEST_F(BackendConfigsTest, ParseAllReduceIsAsync) {
  const absl::string_view kModuleStr = R"(
    HloModule test

    apply_op {
      x = u32[] parameter(0)
      y = u32[] parameter(1)
      ROOT apply_op = u32[] add(x, y)
    }

    ENTRY test_computation {
      id = u32[] replica-id()
      start1 = u32[] all-reduce-start(id), to_apply=apply_op, backend_config={"collective_backend_config": {"is_sync":false}}
      done1 = u32[] all-reduce-done(start1)
      start2 = u32[] all-reduce-start(done1), to_apply=apply_op, backend_config={"collective_backend_config": {"is_sync":true}}
      done2 = u32[] all-reduce-done(start2)
      start3 = u32[] all-reduce-start(done2), to_apply=apply_op
      done3 = u32[] all-reduce-done(start3)
      ROOT done = (u32[], u32[], u32[]) tuple(done1, done2, done3)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto start1 =
      m->entry_computation()->root_instruction()->operand(0)->operand(0);
  auto start2 =
      m->entry_computation()->root_instruction()->operand(1)->operand(0);
  auto start3 =
      m->entry_computation()->root_instruction()->operand(2)->operand(0);
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                          start1->backend_config<GpuBackendConfig>());
  EXPECT_FALSE(backend_config.collective_backend_config().is_sync());
  TF_ASSERT_OK_AND_ASSIGN(backend_config,
                          start2->backend_config<GpuBackendConfig>());
  EXPECT_TRUE(backend_config.collective_backend_config().is_sync());
  TF_ASSERT_OK_AND_ASSIGN(backend_config,
                          start3->backend_config<GpuBackendConfig>());
  EXPECT_FALSE(backend_config.collective_backend_config().is_sync());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
