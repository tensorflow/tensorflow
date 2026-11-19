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

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/platform_util.h"
#include "xla/side_effect_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {
namespace {

using ::testing::UnorderedElementsAre;

class LocalityDomainComputeTest
    : public HloInterpreterReferenceMixin<HloTestBase> {
 protected:
  void SetUp() override {
    HloInterpreterReferenceMixin<HloTestBase>::SetUp();
    absl::StatusOr<se::Platform*> platform = PlatformUtil::GetPlatform("gpu");
    CHECK_OK(platform.status());
    absl::StatusOr<se::StreamExecutor*> executor =
        (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor.status());
    if ((*executor)->GetDeviceDescription().locality_domain_count() < 2) {
      GTEST_SKIP() << "Requires a GPU with at least two locality domains";
    }
  }
};

// Two dependent calls on different locality domains. The second consumes the
// first's result, so the domains synchronize through default memory.
constexpr absl::string_view kTwoDomainCalls = R"(
  HloModule m

  body {
    p0 = f32[1024] parameter(0)
    p1 = f32[1024] parameter(1)
    add = f32[1024] add(p0, p1)
    ROOT multiply = f32[1024] multiply(add, p0)
  }

  ENTRY main {
    p0 = f32[1024] parameter(0)
    p1 = f32[1024] parameter(1)
    call0 = f32[1024] call(p0, p1), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:0",inlineable="false"}
    call1 = f32[1024] call(call0, p1), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:1",inlineable="false"}
    ROOT negate = f32[1024] negate(call1)
  })";

TEST_F(LocalityDomainComputeTest, AnnotatedCallsBecomeLocalityDomainRegions) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       GetOptimizedModule(kTwoDomainCalls));
  std::vector<std::string> annotations;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kAsyncStart) {
        continue;
      }
      auto it =
          instr->frontend_attributes().map().find(kXlaStreamAnnotationAttr);
      if (it != instr->frontend_attributes().map().end()) {
        annotations.push_back(it->second);
      }
    }
  }
  EXPECT_THAT(annotations,
              UnorderedElementsAre("locality_domain:0", "locality_domain:1"));
}

TEST_F(LocalityDomainComputeTest, AnnotatedCallsProduceCorrectResults) {
  EXPECT_TRUE(RunAndCompare(kTwoDomainCalls, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla::gpu
