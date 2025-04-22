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

#include "xla/service/cpu_gpu_shape_verifier.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

class CpuGpuShapeVerifierTest : public HloHardwareIndependentTestBase {
 public:
  CpuGpuShapeVerifierTest() {
    // Create HloVerifier which uses CpuGpuShapeVerifier
    HloVerifierOpts opts;
    std::unique_ptr<TargetVerifierMetadata> metadata =
        std::make_unique<CpuGpuVerifierMetadata>(std::move(opts));
    set_hlo_verifier(std::make_unique<HloVerifier>(std::move(metadata)));
  }
};

TEST_F(CpuGpuShapeVerifierTest, InvalidElementSize) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY main {
    p0 = u8[2,5]{1,0:E(8)} parameter(0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("The XLA CPU/GPU backend does not support custom "
                        "element sizes on non-sub-byte types"));
}

TEST_F(CpuGpuShapeVerifierTest, Int4SupportedInstruction) {
  const char* const hlo_string = R"(
  HloModule Module

  select_bcast {
    p0 = u4[] parameter(0)
    p1 = u4[] reshape(p0)
    p2 = u4[] parameter(1)
    cmp = pred[] compare(p1, p2), direction=LT
    sel = u4[] select(cmp, p1, p2)
    ROOT out = u4[3, 3] broadcast(sel), dimensions={}
  }

  ENTRY main {
    p0 = u4[] parameter(0)
    p1 = u4[] parameter(1)
    ROOT out = u4[3, 3] call(p0, p1), to_apply=select_bcast
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  TF_EXPECT_OK(status);
}

TEST_F(CpuGpuShapeVerifierTest, Int4ShardingCustomCall) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY main {
    p0 = u4[] parameter(0)
    ROOT sharded = u4[] custom-call(p0), custom_call_target="Sharding"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  TF_EXPECT_OK(status);
}

}  // namespace
}  // namespace xla
