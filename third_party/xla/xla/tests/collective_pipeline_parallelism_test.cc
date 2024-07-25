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
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

// Tests cross-GPU operations.
//
// Several tests requires at least four GPUs.  For instructions on running this
// within Google, see go/multi-gpu-unit-test.

// TODO: Move this to hlo_test_base.h
#define SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(x)                     \
  if (num_devices_ < x) {                                         \
    GTEST_SKIP() << "Test requires at least " << x << " devices"; \
  }

namespace xla {
namespace {

class CollectivePipelineParallelismTest : public HloTestBase {
 public:
  CollectivePipelineParallelismTest() : num_devices_(backend().device_count()) {
    VLOG(1) << "Running with " << num_devices_ << " devices";
  }

 protected:
  const int64_t num_devices_;
};

XLA_TEST_F(CollectivePipelineParallelismTest,
           CollectivePermute_CircularPipelinePreOptimization) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  while_cond {
    param = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(param), index=0
    max_iter = u32[] constant(3)
    ROOT cmp = pred[] compare(iter, max_iter), direction=LT
  }

  while_body {
    param = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(param), index=0
    data = f32[2,2] get-tuple-element(param), index=1
    weights = f32[2,2] get-tuple-element(param), index=2
    matmul = f32[2,2] dot(weights, data), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    cp = f32[2,2] collective-permute(matmul), source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
    iter_increment = u32[] constant(1)
    next_iter = u32[] add(iter, iter_increment)
    ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, cp, weights)
  }

  ENTRY test_computation {
    iter = u32[] constant(0)
    data = f32[2,2] parameter(0)
    weights = f32[2,2] parameter(1)
    input = (u32[], f32[2,2], f32[2,2]) tuple(iter, data, weights)
    while_res = (u32[], f32[2,2], f32[2,2]) while(input), condition=while_cond, body=while_body
    ROOT data_out = f32[2,2] get-tuple-element(while_res), index=1
  }
  )";
  const int64_t kNumReplicas = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas)

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  std::unique_ptr<VerifiedHloModule> module;
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  // Input for replica i is
  // {{i, i},
  //  {i, i}}.
  std::vector<Literal> replica_inputs;
  for (float i = 1; i < kNumReplicas + 1; ++i) {
    replica_inputs.push_back({LiteralUtil::CreateR2<float>({{i, i}, {i, i}})});
    replica_inputs.push_back(LiteralUtil::CreateR2<float>({{0, 0}, {0, 1}}));
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          test_runner_.CreateExecutable(
                              std::unique_ptr<HloModule>(std::move(module)),
                              /*run_hlo_passes=*/true));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(
          /*executable_provider=*/[&](int64_t) { return executable.get(); },
          /*argument_count_provider=*/[](int64_t) { return 2; },
          /*argument_provider=*/
          [&](int64_t replica, int64_t index) -> const Literal* {
            return &replica_inputs[replica * 2 + index];
          },
          kNumReplicas, /*run_hlo_passes=*/true,
          /*device_assignment=*/nullptr));
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {2, 2}}, results[0]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {3, 3}}, results[1]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {4, 4}}, results[2]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {1, 1}}, results[3]);
}

}  // namespace
}  // namespace xla
