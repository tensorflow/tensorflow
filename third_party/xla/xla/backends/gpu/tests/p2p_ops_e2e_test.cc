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
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class P2POps : public CollectiveOpsWithFlagsBase,
               public ::testing::WithParamInterface<bool> {
 public:
  P2POps()
      : CollectiveOpsWithFlagsBase(
            /*enable_async=*/true,
            /*enable_p2p_memcpy=*/false,
            /*enable_symmetric_buffer=*/GetParam(),
            /*memory_size=*/32 * kMB,
            /*collectives_memory_size=*/0) {}
};

TEST_P(P2POps, CollectivePermute) {
  const int64_t kNumReplicas = 2;
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  constexpr absl::string_view kModuleStr = R"(
    HloModule p2p_exchange, replica_count=2

    ENTRY main {
      c21 = u32[] constant(21)
      data = u32[2] broadcast(c21), dimensions={}
      ROOT result = u32[2] collective-permute(data), source_target_pairs={{0,1},{1,0}}
    }
  )";
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({21, 21}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({21, 21}, results[1]);
}

INSTANTIATE_TEST_SUITE_P(P2POps, P2POps, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "enable_symmetric_buffer"
                                             : "disable_symmetric_buffer";
                         });

}  // namespace
}  // namespace xla
