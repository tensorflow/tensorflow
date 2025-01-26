/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/topk_specializer.h"

#include <stddef.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/platform_util.h"
#include "xla/service/topk_rewriter.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::Combine;
using ::testing::Values;

// Params:
//  - n_kb: number of elements in kilobytes.
//  - k: number of elements to return.
//  - batch_size
//  - dtype
using ParameterizedInterface =
    ::testing::WithParamInterface<std::tuple<int, int, int, absl::string_view>>;

class TopkTest : public HloTestBase, public ParameterizedInterface {
 public:
  TopkTest()
      : HloTestBase(*PlatformUtil::GetPlatform("gpu"),
                    *PlatformUtil::GetPlatform("gpu"), true, true, {}) {}

 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> TopkHlo(int n, int k,
                                                     int batch_size,
                                                     absl::string_view dtype) {
    return ParseAndReturnVerifiedModule(absl::Substitute(
        R"(
      %compare {
        %p.1.lhs.40628 = s32[] parameter(2)
        %p.1.rhs.40629 = s32[] parameter(3)
        %constant.40630 = pred[] constant(true)
        %broadcast.40631 = pred[] broadcast(pred[] %constant.40630), dimensions={}
        %p.0.lhs.40626 = f32[] parameter(0)
        %p.0.rhs.40627 = f32[] parameter(1)
        %compare.40632 = pred[] compare(f32[] %p.0.lhs.40626, f32[] %p.0.rhs.40627), direction=GT, type=TOTALORDER
        ROOT %select.40633 = pred[] select(pred[] %broadcast.40631, pred[] %compare.40632, pred[] %broadcast.40631)
      }

      ENTRY top_k {
        %arg = $3[$2,$0] parameter(0)
        ROOT %result = ($3[$2,$1], s32[$2,$1]) custom-call(%arg), custom_call_target="TopK", to_apply=%compare
      }
    )",
        n, k, batch_size, dtype));
  }
};

class GeneralizeTopkVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleCustomCall(HloInstruction* inst) override {
    HloCustomCallInstruction* topk = DynCast<HloCustomCallInstruction>(inst);
    if (topk == nullptr || topk->custom_call_target() != "__gpu$TopK") {
      return absl::OkStatus();
    }
    HloComputation* comp = topk->parent();
    auto original_shape = ShapeUtil::SliceTuple(topk->shape(), 0, 2);
    HloInstruction* original_topk =
        comp->AddInstruction(HloInstruction::CreateCustomCall(
            original_shape, topk->operands(), topk->to_apply(), "TopK"));
    // TupleUtil::ExtractPrefix creates the following structure:
    //      TopK
    //   -------------
    //   |     |     |
    //  Get   Get   Get
    //    \    |     /
    //     CreateTuple
    // Here we walk to Create Tuple and replace it with the original topk.
    HloInstruction* new_tuple = topk->users()[0]->users()[0];
    return ReplaceInstruction(new_tuple, original_topk);
  }
};

class GeneralizeTopk : public HloModulePass {
 public:
  absl::string_view name() const override { return "generalized-topk"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    return GeneralizeTopkVisitor().RunOnModule(module, execution_threads);
  }
};

void ToSortAndSlice(HloModule* module) {
  TF_ASSERT_OK_AND_ASSIGN(bool changed, GeneralizeTopk().Run(module));
  ASSERT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(changed, TopkDecomposer().Run(module));
  ASSERT_TRUE(changed);
}

TEST_P(TopkTest, ProducesCorrectResult) {
  const auto [n_kb, k, batch_size, dtype] = GetParam();
  const size_t n = n_kb * 1024;
  TF_ASSERT_OK_AND_ASSIGN(auto topk_module, TopkHlo(n, k, batch_size, dtype));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          gpu::TopkSpecializer().Run(topk_module.get()));
  ASSERT_TRUE(changed);
  EXPECT_TRUE(
      RunAndCompare(std::move(topk_module), std::nullopt, ToSortAndSlice));
}

INSTANTIATE_TEST_SUITE_P(
    TopkTests, TopkTest,
    Combine(
        /*n_kb=*/Values(1, 8, 12, 32),
        /*k=*/Values(1, 2, 4, 8, 16, 7, 12),
        /*batch_size=*/Values(1, 16, 32, 64, 128),
        /*dtype=*/Values(absl::string_view("f32"), "bf16")),
    [](const auto& info) {
      return absl::Substitute("n$0KiB_k$1_batch_size$2_$3",
                              std::get<0>(info.param), std::get<1>(info.param),
                              std::get<2>(info.param), std::get<3>(info.param));
    });

}  // namespace
}  // namespace xla
