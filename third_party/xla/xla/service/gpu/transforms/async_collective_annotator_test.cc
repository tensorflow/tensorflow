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

#include "xla/service/gpu/transforms/async_collective_annotator.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  addf32 {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  addf16 {
    p0 = f16[] parameter(0)
    p1 = f16[] parameter(1)
    ROOT add = f16[] add(p0, p1)
  }

  reduce_scatterf32 {
    p0 = f32[2] parameter(0)
    ROOT result = f32[1] reduce-scatter(p0), replica_groups={},
                      dimensions={0}, to_apply=addf32
  }

  ENTRY entry {
    pf32 = f32[1] parameter(0)
    pf16 = f16[1] parameter(1)

    arf32-start = f32[1] all-reduce-start(pf32), to_apply=addf32
    arf32-done = f32[1] all-reduce-done(arf32-start)

    arf16-start = f16[1] all-reduce-start(pf16), to_apply=addf16
    arf16-done = f16[1] all-reduce-done(arf16-start)

    agf32-start = (f32[1], f32[2]) all-gather-start(pf32), dimensions={0}
    agf32-done = f32[2] all-gather-done(agf32-start)

    agf16-start = (f16[1], f16[2]) all-gather-start(pf16), dimensions={0}
    agf16-done = f16[2] all-gather-done(agf16-start)

    cpf32-start = (f32[1], f32[1], u32[], u32[]) collective-permute-start(pf32),
                    source_target_pairs={{0,1}, {1,0}}
    cpf32-done = f32[1] collective-permute-done(cpf32-start)

    cpf16-start = (f16[1], f16[1], u32[], u32[]) collective-permute-start(pf16),
                    source_target_pairs={{0,1}, {1,0}}
    cpf16-done = f16[1] collective-permute-done(cpf16-start)

    rsf32-start = ((f32[2]), f32[1]) async-start(agf32-done), calls=reduce_scatterf32
    rsf32-done = f32[1] async-done(rsf32-start), calls=reduce_scatterf32

    ROOT tuple = (f32[1], f16[1], f32[2], f16[2], f32[1], f16[1], f32[1])
                tuple(arf32-done, arf16-done, agf32-done, agf16-done, cpf32-done,
                      cpf16-done, rsf32-done)
  }
)";

struct TestCase {
  std::string test_name;
  HloPredicate is_async_predicate;
  absl::flat_hash_set<absl::string_view> expected_async;
  absl::flat_hash_set<absl::string_view> expected_sync;
};

class AsyncCollectiveAnnotatorTest
    : public HloTestBase,
      public ::testing::WithParamInterface<TestCase> {};

XLA_TEST_P(AsyncCollectiveAnnotatorTest, Test) {
  const TestCase& test_case = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloString, /*replica_count=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      AsyncCollectiveAnnotator(test_case.is_async_predicate).Run(module.get()));
  EXPECT_TRUE(changed);

  // Assert that all async collectives are annotated with the backend config.
  for (const HloInstruction* hlo :
       module->entry_computation()->instructions()) {
    if (!hlo_query::IsAsyncCollectiveStartOp(hlo)) {
      continue;
    }
    auto gpu_config = hlo->backend_config<GpuBackendConfig>();
    ASSERT_TRUE(gpu_config.ok());

    const CollectiveBackendConfig& backend_config =
        gpu_config.value().collective_backend_config();
    if (test_case.expected_async.contains(hlo->name())) {
      EXPECT_FALSE(backend_config.is_sync());
    }

    if (test_case.expected_sync.contains(hlo->name())) {
      EXPECT_TRUE(backend_config.is_sync());
    }
  }
}

std::vector<TestCase> TestCases() {
  HloPredicate is_f16 = [](const HloInstruction* hlo) {
    return hlo->operand(0)->shape().element_type() == PrimitiveType::F16;
  };

  return {
      {"all_async",
       HloPredicateTrue, /*expected_async=*/
       {"arf32-start", "arf16-start", "agf32-start", "agf16-start",
        "cpf32-start", "cpf16-start", "rsf32-start"},
       /*expected_sync=*/{}},
      {"all_sync",
       HloPredicateFalse,
       /*expected_async=*/{},
       /*expected_sync=*/
       {"arf32-start", "arf16-start", "agf32-start", "agf16-start",
        "cpf32-start", "cpf16-start", "rsf32-start"}},
      {"ar_async",
       HloPredicateIsOp<HloOpcode::kAllReduceStart>,
       /*expected_async=*/
       {"arf32-start", "arf16-start"},
       /*expected_sync=*/
       {"agf32-start", "agf16-start", "cpf32-start", "cpf16-start",
        "rsf32-start"}},
      {"cp_async",
       HloPredicateIsOp<HloOpcode::kCollectivePermuteStart>,
       /*expected_async=*/
       {"cpf32-start", "cpf16-start"},
       /*expected_sync=*/
       {"arf32-start", "arf16-start", "agf32-start", "agf16-start",
        "rsf32-start"}},
      {"f16_async",
       is_f16,
       /*expected_async=*/{"arf16-start", "agf16-start", "cpf16-start"},
       /*expected_sync=*/
       {"arf32-start", "agf32-start", "cpf32-start", "rsf32-start"}},
  };
}

std::string TestCaseName(const ::testing::TestParamInfo<TestCase>& test_case) {
  return test_case.param.test_name;
}

INSTANTIATE_TEST_SUITE_P(AsyncCollectiveAnnotatorTest,
                         AsyncCollectiveAnnotatorTest,
                         ::testing::ValuesIn(TestCases()), TestCaseName);
}  // namespace
}  // namespace gpu
}  // namespace xla
