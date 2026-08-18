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

#include "xla/backends/gpu/transforms/collectives/legalize_collective_domain.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/transforms/collectives/collective_domain.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class LegalizeCollectiveDomainTest : public HloHardwareIndependentTestBase {};

TEST(CollectiveDomainTest, StringRoundTrip) {
  for (CollectiveCommunicationDomain domain :
       {kUnspecifiedCollectiveDomain, kScaleUpFabricCollectiveDomain}) {
    EXPECT_THAT(
        ParseCollectiveCommunicationDomain(absl::StrFormat("%v", domain)),
        absl_testing::IsOkAndHolds(domain));
  }
}

TEST(CollectiveDomainTest, ParsingIsCaseInsensitive) {
  EXPECT_THAT(ParseCollectiveCommunicationDomain("UnSpEcIfIeD"),
              absl_testing::IsOkAndHolds(kUnspecifiedCollectiveDomain));
  EXPECT_THAT(ParseCollectiveCommunicationDomain("ScAlE_Up_FaBrIc"),
              absl_testing::IsOkAndHolds(kScaleUpFabricCollectiveDomain));
}

TEST_F(LegalizeCollectiveDomainTest, PromotesFrontendAttribute) {
  const char* const hlo = R"(
    HloModule m, replica_count=2

    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT sum = f32[] add(x, y)
    }

    ENTRY main {
      p0 = f32[1] parameter(0)
      ROOT ar = f32[1] all-reduce(p0), replica_groups={{0,1}},
        to_apply=add,
        frontend_attributes={collective_communication_domain="scale_up_fabric"}
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  LegalizeCollectiveDomain legalizer;
  EXPECT_THAT(legalizer.Run(module.get()), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
//  CHECK-NOT: collective_communication_domain
//      CHECK: ROOT %ar = {{.*}}all-reduce({{.*}})
// CHECK-SAME:   backend_config={{.*}}communication_domain
// CHECK-SAME:   COLLECTIVE_COMMUNICATION_DOMAIN_SCALE_UP_FABRIC
//  CHECK-NOT: collective_communication_domain
)"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(legalizer.Run(module.get()), absl_testing::IsOkAndHolds(false));
}

TEST_F(LegalizeCollectiveDomainTest, AcceptsBackendConfig) {
  const char* const hlo = R"(
    HloModule m

    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT sum = f32[] add(x, y)
    }

    ENTRY main {
      p0 = f32[1] parameter(0)
      ROOT ar = f32[1] all-reduce(p0), to_apply=add
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  HloInstruction* ar = FindInstruction(module.get(), "ar");
  GpuBackendConfig config;
  config.mutable_collective_backend_config()->set_communication_domain(
      kScaleUpFabricCollectiveDomain);
  ASSERT_OK(ar->set_backend_config(config));

  EXPECT_THAT(LegalizeCollectiveDomain().Run(module.get()),
              absl_testing::IsOkAndHolds(false));
  EXPECT_FALSE(ar->get_frontend_attribute(kCollectiveCommunicationDomainAttr)
                   .has_value());

  config.mutable_collective_backend_config()->set_communication_domain(
      static_cast<CollectiveCommunicationDomain>(7));
  ASSERT_OK(ar->set_backend_config(config));
  EXPECT_THAT(
      LegalizeCollectiveDomain().Run(module.get()),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unknown(7)")));
}

TEST_F(LegalizeCollectiveDomainTest, RejectsUnknownDomain) {
  const char* const hlo = R"(
    HloModule m

    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT sum = f32[] add(x, y)
    }

    ENTRY main {
      p0 = f32[1] parameter(0)
      ROOT ar = f32[1] all-reduce(p0), to_apply=add,
        frontend_attributes={collective_communication_domain="local"}
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  EXPECT_THAT(LegalizeCollectiveDomain().Run(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("communication domain: local")));
}

TEST_F(LegalizeCollectiveDomainTest, RejectsAttributeOnNonCollective) {
  const char* const hlo = R"(
    HloModule m

    ENTRY main {
      p0 = f32[1] parameter(0),
        frontend_attributes={collective_communication_domain="scale_up_fabric"}
      ROOT copy = f32[1] copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  EXPECT_THAT(LegalizeCollectiveDomain().Run(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not supported on p0")));
}

TEST_F(LegalizeCollectiveDomainTest, PromotesLegacyAsyncCollective) {
  const char* const hlo = R"(
    HloModule m

    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT sum = f32[] add(x, y)
    }

    ENTRY main {
      p0 = f32[1] parameter(0)
      start = f32[1] all-reduce-start(p0),
        to_apply=add,
        frontend_attributes={collective_communication_domain="scale_up_fabric"}
      ROOT done = f32[1] all-reduce-done(start)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  LegalizeCollectiveDomain legalizer;
  EXPECT_THAT(legalizer.Run(module.get()), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
//  CHECK-NOT: collective_communication_domain
//      CHECK: %start = {{.*}}all-reduce-start({{.*}})
// CHECK-SAME:   backend_config={{.*}}communication_domain
// CHECK-SAME:   COLLECTIVE_COMMUNICATION_DOMAIN_SCALE_UP_FABRIC
// CHECK:      ROOT %done = {{.*}}all-reduce-done(%start)
// CHECK-NOT: collective_communication_domain
)"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(legalizer.Run(module.get()), absl_testing::IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla::gpu
