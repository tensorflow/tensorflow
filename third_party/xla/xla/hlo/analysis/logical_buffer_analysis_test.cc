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

#include "xla/hlo/analysis/logical_buffer_analysis.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace {

using ::testing::UnorderedElementsAre;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

class LogicalBufferAnalysisTest : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<LogicalBufferAnalysis> analysis_;

  // Verifies that a buffer is defined by `instruction` at `index` and matches.
  void VerifyBufferDefinedAt(HloInstruction* instruction,
                             const ShapeIndex& index) {
    ASSERT_OK_AND_ASSIGN(LogicalBuffer * buffer,
                         analysis_->GetBuffer(instruction, index));
    EXPECT_EQ(buffer->instruction(), instruction);
    EXPECT_EQ(buffer->index(), index);
    EXPECT_THAT(analysis_->GetBuffer(buffer->id()), IsOkAndHolds(buffer));
  }

  // Verifies that no buffer is defined by `instruction` at `index`.
  void VerifyNoBufferDefinedAt(const HloInstruction* instruction,
                               const ShapeIndex& index) {
    for (const auto& buf : analysis_->logical_buffers()) {
      if (buf->instruction() == instruction && buf->index() == index) {
        ADD_FAILURE() << "Instruction " << instruction->name() << " at index "
                      << index.ToString()
                      << " should not define a logical buffer.";
      }
    }
  }

  // Returns all defining locations present in the analysis.
  std::vector<std::pair<const HloInstruction*, ShapeIndex>> GetDefiningSites() {
    std::vector<std::pair<const HloInstruction*, ShapeIndex>> defining_sites;
    for (const auto& buf : analysis_->logical_buffers()) {
      defining_sites.push_back({buf->instruction(), buf->index()});
    }
    return defining_sites;
  }
};

TEST_F(LogicalBufferAnalysisTest, BasicAndAccessors) {
  const absl::string_view hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    ROOT const = f32[] constant(1.0)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  HloInstruction* const_inst = FindInstruction(module.get(), "const");
  HloInstruction* param = FindInstruction(module.get(), "p0");

  EXPECT_EQ(analysis_->num_logical_buffers(), 2);
  EXPECT_FALSE(analysis_->logical_buffers().empty());

  VerifyBufferDefinedAt(const_inst, {});
  VerifyBufferDefinedAt(param, {});

  EXPECT_THAT(GetDefiningSites(),
              UnorderedElementsAre(std::make_pair(const_inst, ShapeIndex({})),
                                   std::make_pair(param, ShapeIndex({}))));
}

TEST_F(LogicalBufferAnalysisTest, DataflowOps) {
  absl::string_view hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    p1 = f32[2,3] parameter(1)
    tok = token[] after-all()
    tuple = (f32[2,3], f32[2,3]) tuple(p0, p1)
    gte = f32[2,3] get-tuple-element(tuple), index=0
    bitcast = f32[2,3] bitcast(gte)
    domain_inst = f32[2,3] domain(bitcast), domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
    copy = f32[2,3] copy(domain_inst)
    ROOT dep = f32[2,3] add-dependency(copy, tok)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* p1 = FindInstruction(module.get(), "p1");
  HloInstruction* tok = FindInstruction(module.get(), "tok");
  HloInstruction* tuple = FindInstruction(module.get(), "tuple");
  HloInstruction* gte = FindInstruction(module.get(), "gte");
  HloInstruction* bitcast = FindInstruction(module.get(), "bitcast");
  HloInstruction* domain_inst = FindInstruction(module.get(), "domain_inst");
  HloInstruction* copy = FindInstruction(module.get(), "copy");
  HloInstruction* dep = FindInstruction(module.get(), "dep");

  EXPECT_EQ(analysis_->num_logical_buffers(), 5);

  VerifyBufferDefinedAt(p0, {});
  VerifyBufferDefinedAt(p1, {});
  VerifyBufferDefinedAt(tok, {});
  VerifyBufferDefinedAt(tuple, {});
  VerifyBufferDefinedAt(copy, {});

  VerifyNoBufferDefinedAt(tuple, {0});
  VerifyNoBufferDefinedAt(tuple, {1});
  VerifyNoBufferDefinedAt(gte, {});
  VerifyNoBufferDefinedAt(bitcast, {});
  VerifyNoBufferDefinedAt(domain_inst, {});
  VerifyNoBufferDefinedAt(dep, {});

  EXPECT_THAT(GetDefiningSites(),
              UnorderedElementsAre(std::make_pair(p0, ShapeIndex({})),
                                   std::make_pair(p1, ShapeIndex({})),
                                   std::make_pair(tok, ShapeIndex({})),
                                   std::make_pair(tuple, ShapeIndex({})),
                                   std::make_pair(copy, ShapeIndex({}))));
}

TEST_F(LogicalBufferAnalysisTest, CopyStartDone) {
  absl::string_view hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    copy-start = (f32[2,3], f32[2,3], u32[]) copy-start(p0)
    ROOT copy-done = f32[2,3] copy-done(copy-start)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* copy_start = FindInstruction(module.get(), "copy-start");
  HloInstruction* copy_done = FindInstruction(module.get(), "copy-done");

  EXPECT_EQ(analysis_->num_logical_buffers(), 4);

  VerifyBufferDefinedAt(p0, {});
  VerifyBufferDefinedAt(copy_start, {});
  VerifyBufferDefinedAt(copy_start, {0});
  VerifyBufferDefinedAt(copy_start, {2});

  VerifyNoBufferDefinedAt(copy_start, {1});
  VerifyNoBufferDefinedAt(copy_done, {});

  EXPECT_THAT(
      GetDefiningSites(),
      UnorderedElementsAre(std::make_pair(p0, ShapeIndex({})),
                           std::make_pair(copy_start, ShapeIndex({})),
                           std::make_pair(copy_start, ShapeIndex({0})),
                           std::make_pair(copy_start, ShapeIndex({2}))));
}

TEST_F(LogicalBufferAnalysisTest, SendRecvDone) {
  absl::string_view hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    tok = token[] after-all()
    send = (f32[2,3], u32[], token[]) send(p0, tok), channel_id=1
    send-done = token[] send-done(send), channel_id=1
    recv = (f32[2,3], u32[], token[]) recv(tok), channel_id=2
    ROOT recv-done = (f32[2,3], token[]) recv-done(recv), channel_id=2
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* tok = FindInstruction(module.get(), "tok");
  HloInstruction* send = FindInstruction(module.get(), "send");
  HloInstruction* send_done = FindInstruction(module.get(), "send-done");
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  HloInstruction* recv_done = FindInstruction(module.get(), "recv-done");

  EXPECT_EQ(analysis_->num_logical_buffers(), 12);

  VerifyBufferDefinedAt(p0, {});
  VerifyBufferDefinedAt(tok, {});
  VerifyBufferDefinedAt(send, {});
  VerifyBufferDefinedAt(send, {1});
  VerifyBufferDefinedAt(send, {2});
  VerifyBufferDefinedAt(send_done, {});

  VerifyBufferDefinedAt(recv, {});
  VerifyBufferDefinedAt(recv, {0});
  VerifyBufferDefinedAt(recv, {1});
  VerifyBufferDefinedAt(recv, {2});

  VerifyBufferDefinedAt(recv_done, {});
  VerifyBufferDefinedAt(recv_done, {1});

  VerifyNoBufferDefinedAt(send, {0});
  VerifyNoBufferDefinedAt(recv_done, {0});

  EXPECT_THAT(GetDefiningSites(),
              UnorderedElementsAre(std::make_pair(p0, ShapeIndex({})),
                                   std::make_pair(tok, ShapeIndex({})),
                                   std::make_pair(send, ShapeIndex({})),
                                   std::make_pair(send, ShapeIndex({1})),
                                   std::make_pair(send, ShapeIndex({2})),
                                   std::make_pair(send_done, ShapeIndex({})),
                                   std::make_pair(recv, ShapeIndex({})),
                                   std::make_pair(recv, ShapeIndex({0})),
                                   std::make_pair(recv, ShapeIndex({1})),
                                   std::make_pair(recv, ShapeIndex({2})),
                                   std::make_pair(recv_done, ShapeIndex({})),
                                   std::make_pair(recv_done, ShapeIndex({1}))));
}

TEST_F(LogicalBufferAnalysisTest, CustomCallFusionAsync) {
  absl::string_view hlo_str = R"(
  HloModule module

  fused_computation {
    p0.fused = f32[2,3] parameter(0)
    ROOT copy.fused = f32[2,3] copy(p0.fused)
  }

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    ccall = f32[2,3] custom-call(p0), custom_call_target="cc_target",
            output_to_operand_aliasing={ {}: (0, {}) }
    fusion = f32[2,3] fusion(p0), kind=kLoop, calls=fused_computation
    async-start = ((f32[2,3]), f32[2,3], u32[]) custom-call-start(p0),
                                                custom_call_target="bar"
    async-update = ((f32[2,3]), f32[2,3], u32[]) async-update(async-start)
    ROOT async-done = f32[2,3] custom-call-done(async-update)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  HloInstruction* p0_fused = FindInstruction(module.get(), "p0.fused");
  HloInstruction* copy_fused = FindInstruction(module.get(), "copy.fused");
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* ccall = FindInstruction(module.get(), "ccall");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");
  HloInstruction* async_start = FindInstruction(module.get(), "async-start");
  HloInstruction* async_update = FindInstruction(module.get(), "async-update");
  HloInstruction* async_done = FindInstruction(module.get(), "async-done");
  HloComputation* async_computation = async_start->async_wrapped_computation();
  HloInstruction* async_wrapper_root = async_computation->root_instruction();
  ASSERT_EQ(async_wrapper_root->operand_count(), 1);
  HloInstruction* async_wrapper_param = async_wrapper_root->mutable_operand(0);

  EXPECT_EQ(analysis_->num_logical_buffers(), 13);

  VerifyBufferDefinedAt(p0, {});
  VerifyBufferDefinedAt(ccall, {});
  VerifyBufferDefinedAt(fusion, {});
  VerifyBufferDefinedAt(p0_fused, {});
  VerifyBufferDefinedAt(copy_fused, {});

  VerifyBufferDefinedAt(async_start, {});
  VerifyBufferDefinedAt(async_start, {0});
  VerifyBufferDefinedAt(async_start, {1});
  VerifyBufferDefinedAt(async_start, {2});
  // {0, 0} is implicitly aliased to input parameter p0.
  VerifyNoBufferDefinedAt(async_start, {0, 0});

  VerifyBufferDefinedAt(async_update, {});
  VerifyNoBufferDefinedAt(async_update, {0});
  VerifyNoBufferDefinedAt(async_update, {1});
  VerifyNoBufferDefinedAt(async_update, {2});

  // There are two more buffers defined in the async computation,
  // which is created by the custom-call-start and custom-call-done
  // in async wrapped computation.
  VerifyBufferDefinedAt(async_wrapper_param, {});
  VerifyBufferDefinedAt(async_wrapper_root, {});

  VerifyBufferDefinedAt(async_done, {});
}

TEST_F(LogicalBufferAnalysisTest, NestedAndAliasedFusion) {
  absl::string_view hlo_str = R"(
  HloModule module

  fused_computation_inner {
    p0.inner = f32[2,3] parameter(0)
    ROOT copy.inner = f32[2,3] copy(p0.inner)
  }

  fused_computation_outer {
    p0.outer = f32[2,3] parameter(0)
    ROOT fusion.inner = f32[2,3] fusion(p0.outer), kind=kLoop,
                                 calls=fused_computation_inner
  }

  fused_computation_alias {
    p0.alias = f32[2,3] parameter(0)
    ROOT copy.alias = f32[2,3] copy(p0.alias)
  }

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    fusion.outer = f32[2,3] fusion(p0), kind=kLoop, calls=fused_computation_outer
    ROOT fusion.alias = f32[2,3] fusion(p0), kind=kLoop,
                                 calls=fused_computation_alias,
                                 output_to_operand_aliasing={ {}: (0, {}) }
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* fusion_outer = FindInstruction(module.get(), "fusion.outer");

  HloInstruction* fusion_alias = FindInstruction(module.get(), "fusion.alias");

  VerifyBufferDefinedAt(p0, {});
  VerifyBufferDefinedAt(fusion_outer, {});
  // No buffer is defined at the output of fusion.alias because it is explicitly
  // aliased to the input.
  VerifyNoBufferDefinedAt(fusion_alias, {});
}

TEST_F(LogicalBufferAnalysisTest, AsyncStartWithOutputToOperandAliasing) {
  absl::string_view hlo_str = R"(
  HloModule module

  fused_computation_async {
    p0.async = f32[2,3] parameter(0)
    ROOT copy.async = f32[2,3] copy(p0.async)
  }

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[2,3], u32[]) async-start(p0),
                  calls=fused_computation_async,
                  output_to_operand_aliasing={ {1}: (0, {}) }
    ROOT async-done = f32[2,3] async-done(async-start)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* async_start = FindInstruction(module.get(), "async-start");

  VerifyBufferDefinedAt(p0, {});

  // async-start defines output except implicitly/explicitly aliased:
  // - {} (top-level tuple) -> defined (not aliased)
  // - {0} (sub-tuple) -> defined
  // - {0, 0} -> implicitly aliased
  // - {1} -> explicitly aliased (to p0)
  // - {2} (state) -> defined
  VerifyBufferDefinedAt(async_start, {});
  VerifyBufferDefinedAt(async_start, {0});
  VerifyBufferDefinedAt(async_start, {2});

  VerifyNoBufferDefinedAt(async_start, {0, 0});
  VerifyNoBufferDefinedAt(async_start, {1});
}

TEST_F(LogicalBufferAnalysisTest, CustomCallAliasingWithDataflowFlag) {
  absl::string_view hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    ROOT ccall = f32[2,3] custom-call(p0), custom_call_target="cc_target",
                 output_to_operand_aliasing={ {}: (0, {}) }
  }
  )";

  // Case 1: alias_buffer_across_dataflow = false (default)
  {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
    ASSERT_OK_AND_ASSIGN(
        analysis_,
        LogicalBufferAnalysis::Run(module.get(),
                                   /*alias_buffer_across_dataflow=*/false));
    HloInstruction* ccall = FindInstruction(module.get(), "ccall");
    VerifyBufferDefinedAt(ccall, {});
  }

  // Case 2: alias_buffer_across_dataflow = true
  {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
    ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(
                                        module.get(),
                                        /*alias_buffer_across_dataflow=*/true));
    HloInstruction* ccall = FindInstruction(module.get(), "ccall");
    VerifyNoBufferDefinedAt(ccall, {});
  }
}

TEST_F(LogicalBufferAnalysisTest, InvalidGetBufferThrows) {
  const absl::string_view hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    ROOT const = f32[] constant(1.0)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  HloInstruction* param = FindInstruction(module.get(), "p0");

  EXPECT_THAT(analysis_->GetBuffer(param, {0}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(LogicalBufferAnalysisTest, InvalidGetBufferIdBehavior) {
  const absl::string_view hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    ROOT const = f32[] constant(1.0)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  // GetBuffer with invalid ID of 100 on a module containing 2 buffers.
  EXPECT_THAT(analysis_->GetBuffer(100),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(LogicalBufferAnalysisTest, SimpleAsyncChain) {
  absl::string_view hlo_str = R"(
  HloModule module

  async_computation {
    p0 = f32[4] parameter(0)
    ROOT ccall = f32[4] custom-call(p0), custom_call_target="bar"
  }

  ENTRY entry {
    p0 = f32[4] parameter(0)
    start = ((f32[4]), f32[4], s32[]) async-start(p0), calls=async_computation,
                                       async_execution_thread="sparsecore"
    update = ((f32[4]), f32[4], s32[]) async-update(start)
    ROOT done = f32[4] async-done(update)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* start = FindInstruction(module.get(), "start");
  HloInstruction* update = FindInstruction(module.get(), "update");

  VerifyBufferDefinedAt(p0, {});

  // Verify start buffers:
  // - {} (top-level tuple) -> defined
  // - {0} (sub-tuple) -> defined
  // - {0, 0} -> implicitly aliased (to input parameter p0) -> not defined
  // - {1} (output) -> defined
  // - {2} (state) -> defined
  VerifyBufferDefinedAt(start, {});
  VerifyBufferDefinedAt(start, {0});
  VerifyBufferDefinedAt(start, {1});
  VerifyBufferDefinedAt(start, {2});
  VerifyNoBufferDefinedAt(start, {0, 0});

  // Verify update buffers:
  // - {} (top-level tuple) -> defined (always defined for new async-update)
  // - all other subshapes are compatible with start -> not defined
  VerifyBufferDefinedAt(update, {});
  VerifyNoBufferDefinedAt(update, {0});
  VerifyNoBufferDefinedAt(update, {0, 0});
  VerifyNoBufferDefinedAt(update, {1});
  VerifyNoBufferDefinedAt(update, {2});
}

TEST_F(LogicalBufferAnalysisTest, AsyncUpdateWithChangedShapes) {
  absl::string_view hlo_str = R"(
  HloModule module

  async_computation {
    p0 = f32[4] parameter(0)
    ROOT ccall = f32[4] custom-call(p0), custom_call_target="bar"
  }

  ENTRY entry {
    p0 = f32[4] parameter(0)
    start = ((f32[4]), f32[4], s32[]) async-start(p0), calls=async_computation,
                                       async_execution_thread="sparsecore"
    update = ((f32[4]), f32[4], s32[]) async-update(start)
    ROOT done = f32[4] async-done(update)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_str));

  HloInstruction* update = FindInstruction(module.get(), "update");

  // Mutate the shape of update to ((f32[8]), f32[8], s32[])
  // This makes it different from start's shape.
  Shape new_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {8})}),
       ShapeUtil::MakeShape(F32, {8}), ShapeUtil::MakeShape(S32, {})});
  *update->mutable_shape() = new_shape;

  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  // {0, 0} is an input operand (starts with 0, size >= 2).
  // It is NOT compatible with start's {0, 0} (f32[4] vs f32[8]).
  // It should NOT define a buffer because it's an input.
  VerifyNoBufferDefinedAt(update, {0, 0});

  // {1} is an output.
  // It is NOT compatible with start's {1} (f32[4] vs f32[8]).
  // It SHOULD define a buffer because it's a new output.
  VerifyBufferDefinedAt(update, {1});
}

TEST_F(LogicalBufferAnalysisTest,
       AsyncUpdateWithExplicitAliasAndChangedShapes) {
  absl::string_view hlo_str = R"(
  HloModule module

  async_computation {
    p0 = f32[4] parameter(0)
    ROOT ccall = f32[4] custom-call(p0), custom_call_target="bar"
  }

  ENTRY entry {
    p0 = f32[4] parameter(0)
    start = ((f32[4]), f32[4], s32[]) async-start(p0), calls=async_computation,
                                       async_execution_thread="sparsecore",
                                       output_to_operand_aliasing={ {1}: (0, {}) }
    update = ((f32[4]), f32[4], s32[]) async-update(start)
    ROOT done = f32[4] async-done(update)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_str));

  HloInstruction* update = FindInstruction(module.get(), "update");

  // Mutate the shape of update to ((f32[4]), f32[8], s32[])
  // This changes the shape of the aliased output {1} from f32[4] to f32[8].
  Shape new_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4})}),
       ShapeUtil::MakeShape(F32, {8}), ShapeUtil::MakeShape(S32, {})});
  *update->mutable_shape() = new_shape;

  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  // {1} is explicitly aliased.
  // It is NOT compatible with start's {1} (f32[4] vs f32[8]).
  // It should NOT define a buffer because it's aliased.
  VerifyNoBufferDefinedAt(update, {1});
}
TEST_F(LogicalBufferAnalysisTest, AsyncAliasedEmptyTupleBecomesValued) {
  absl::string_view hlo_str = R"(
  HloModule module

  async_computation {
    p0 = (f32[4]) parameter(0)
    ROOT ccall = () custom-call(p0), custom_call_target="bar"
  }

  ENTRY entry {
    p0 = (f32[4]) parameter(0)
    start = (((f32[4])), (), s32[]) async-start(p0), calls=async_computation,
                                       async_execution_thread="sparsecore",
                                       output_to_operand_aliasing={ {1}: (0, {}) }
    update = (((f32[4])), (), s32[]) async-update(start)
    ROOT done = f32[4] async-done(update)
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_str));

  HloInstruction* start = FindInstruction(module.get(), "start");
  HloInstruction* update = FindInstruction(module.get(), "update");

  // Mutate update shape to (((f32[4])), (f32[4]), s32[])
  // This simulates late binding where the empty tuple becomes a non-empty
  // tuple.
  Shape f32_4 = ShapeUtil::MakeShape(F32, {4});
  Shape new_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({ShapeUtil::MakeTupleShape({f32_4})}),
       ShapeUtil::MakeTupleShape({f32_4}), ShapeUtil::MakeShape(S32, {})});
  *update->mutable_shape() = new_shape;

  ASSERT_OK_AND_ASSIGN(analysis_, LogicalBufferAnalysis::Run(module.get()));

  // For start:
  // {1} is () and is aliased. It SHOULD get a buffer because it is empty.
  VerifyBufferDefinedAt(start, {1});

  // For update:
  // {1} is (f32[4]) (non-empty tuple) and is aliased. It SHOULD NOT get a
  // buffer.
  VerifyNoBufferDefinedAt(update, {1});

  // {1, 0} is f32[4] and is NOT aliased. It SHOULD get a buffer.
  VerifyBufferDefinedAt(update, {1, 0});
}

}  // namespace
}  // namespace xla
