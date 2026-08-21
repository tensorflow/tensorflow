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
#include <utility>
#include <vector>

#include "xla/tests/xla_test_backend_predicates.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility_test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

class AotCompatibilityCollectivesTest : public AotCompatibilityTestBase {
 public:
  AotCompatibilityCollectivesTest()
      : AotCompatibilityTestBase(tsl::io::JoinPath(
            tsl::testing::TensorFlowSrcRoot(),
            "compiler/xla/tests/aot_compatibility/artifacts")) {
    VLOG(1) << "Running with " << num_devices() << " devices";
  }

  int64_t num_devices() const { return test_runner().device_count(); }
};

TEST_P(AotCompatibilityCollectivesTest, ReplicaId) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    ROOT out = u32[] copy(id)
  }
  )";

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/num_devices());
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, config));

  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), absl::Span<const Literal* const>{},
                        num_devices(), /*use_threads=*/true,
                        /*run_hlo_passes=*/true));

  ASSERT_EQ(results.size(), num_devices());
  for (uint32_t i = 0; i < num_devices(); ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR0(i), results[i]));
  }
}

TEST_P(AotCompatibilityCollectivesTest, CollectiveBroadcast_TwoGPUs) {
  if (test::DeviceIs(test::kCpu)) {
    GTEST_SKIP() << "Test not supported on CPU.";
  }
  if (test_runner().device_count() < 2) {
    GTEST_SKIP() << "Test requires at least 2 devices.";
  }
  const char* const kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    p = u32[] parameter(0)
    ROOT out = u32[] collective-broadcast(p), replica_groups={{0,1}}
  }
  )";

  HloModuleConfig config = GetModuleConfigForTest(/*replica_count=*/2);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, config));

  Literal arg0 = LiteralUtil::CreateR0<uint32_t>(10);
  Literal arg1 = LiteralUtil::CreateR0<uint32_t>(20);

  std::vector<Literal*> args0 = {&arg0};
  std::vector<Literal*> args1 = {&arg1};
  std::vector<std::vector<Literal*>> args = {args0, args1};

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       ExecuteReplicated(std::move(module), args,
                                         /*num_devices=*/2,
                                         /*run_hlo_passes=*/true));

  ASSERT_EQ(results.size(), 2);
  LiteralTestUtil::ExpectR0Equal<uint32_t>(10, results[0]);
  LiteralTestUtil::ExpectR0Equal<uint32_t>(10, results[1]);
}

INSTANTIATE_TEST_SUITE_P(
    AotCompat, AotCompatibilityCollectivesTest,
    ::testing::ValuesIn(GetAvailableAotVersions(
        tsl::io::JoinPath(tsl::testing::TensorFlowSrcRoot(),
                          "compiler/xla/tests/aot_compatibility/artifacts"))),
    AotTestConfigNameGenerator{});

}  // namespace
}  // namespace xla
