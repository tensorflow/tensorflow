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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility_experimental/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla {
namespace aot_compatibility_experimental {
namespace {

class CollectiveOpsAotTest
    : public AotCompatibilityTest,
      public ::testing::WithParamInterface<AotTestParam> {
 public:
  CollectiveOpsAotTest() : AotCompatibilityTest(GetParam()) {}

 protected:
  std::string file_path_;

  void SetUp() override {
    AotCompatibilityTest::SetUp();
    int version = GetParam().version;
    std::string full_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string test_name = full_name.substr(0, full_name.find('/'));
    file_path_ = tsl::io::JoinPath(
        GetExecutablesDirectory(GetParam().target_name),
        absl::StrCat("v", version), absl::StrCat(test_name, ".pbtxt"));
  }
};

TEST_P(CollectiveOpsAotTest, AllGatherMixedTypes) {
  absl::string_view kModuleStr = R"hlo(
    HloModule test
    ENTRY test_computation {
      id = u32[] replica-id()
      p0 = u32[2, 1] broadcast(id), dimensions={}
      p1 = f32[2, 1] convert(p0)
      allgather = (u32[2, 2], f32[2, 2]) all-gather(p0, p1), dimensions={1}
      ag0 = u32[2, 2] get-tuple-element(allgather), index=0
      ag1 = f32[2, 2] get-tuple-element(allgather), index=1
      r0 = u32[4] reshape(ag0)
      r1 = f32[4] reshape(ag1)
      ROOT out = (u32[4], f32[4]) tuple(r0, r1)
    }
  )hlo";
  const int64_t kNumReplicas = 2;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, config));

  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), absl::Span<Literal* const>{},
                        kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  for (int replica_idx = 0; replica_idx < kNumReplicas; replica_idx++) {
    auto rs = results[replica_idx].DecomposeTuple();
    LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 1, 0, 1}, rs[0]);
    LiteralTestUtil::ExpectR1Near<float>({0.0, 1.0, 0.0, 1.0}, rs[1],
                                         ErrorSpec{1e-5, 1e-5});
  }
}

TEST_P(CollectiveOpsAotTest, ReduceScatter) {
  absl::string_view kModuleStr = R"hlo(
    HloModule test
    add {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT add = u32[] add(lhs, rhs)
    }

    ENTRY main {
      c0 = u32[8] constant({1, 2, 3, 4, 5, 6, 7, 8})
      c1 = u32[8] constant({10, 11, 12, 13, 14, 15, 16, 17})
      zero = u32[] constant(0)
      id = u32[] replica-id()
      p = pred[] compare(id, zero), direction=EQ
      pb = pred[8] broadcast(p), dimensions={}
      // data = c0 for replica 0 and c1 for replica 1
      data = u32[8] select(pb, c0, c1)
      ROOT ars = u32[4] reduce-scatter(data), replica_groups={},
                        dimensions={0}, to_apply=add
    }
  )hlo";

  const int64_t kNumReplicas = 2;
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, config));

  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), absl::Span<Literal* const>{},
                        kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));

  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 13, 15, 17}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({19, 21, 23, 25}, results[1]);
}

std::vector<AotTestParam> GetTestParamsOrDie(
    absl::StatusOr<std::vector<AotTestParam>> params) {
  CHECK(params.ok()) << params.status();
  return *std::move(params);
}

INSTANTIATE_TEST_SUITE_P(
    BackwardsCompatibility, CollectiveOpsAotTest,
    ::testing::ValuesIn(
        GetTestParamsOrDie(GetAotTestParamsForBackwardsCompatibility(
            "collective_ops_aot_test_2gpu"))),
    [](const ::testing::TestParamInfo<AotTestParam>& info) {
      return absl::StrCat("v", info.param.version);
    });

INSTANTIATE_TEST_SUITE_P(
    GoldenFileVerification, CollectiveOpsAotTest,
    ::testing::ValuesIn(
        GetTestParamsOrDie(GetAotTestParamsForGoldenFileVerification(
            "collective_ops_aot_test_2gpu"))),
    [](const ::testing::TestParamInfo<AotTestParam>& info) {
      return absl::StrCat("v", info.param.version);
    });

}  // namespace
}  // namespace aot_compatibility_experimental
}  // namespace xla
