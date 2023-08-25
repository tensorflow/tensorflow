/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::IsFalse;
using ::tsl::testing::IsOk;

using BackendConfigsTest = HloTestBase;

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

  const HloInstruction *ags = FindInstruction(module.get(), "agf32-start");
  EXPECT_THAT(ags->has_backend_config(), IsFalse());
  auto collective_backend_config =
      ags->backend_config<CollectiveBackendConfig>();
  EXPECT_THAT(collective_backend_config.status(), IsOk());
  EXPECT_THAT(collective_backend_config->is_sync(), IsFalse());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
