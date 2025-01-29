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

#include "xla/service/collective_permute_decomposer.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

namespace op = xla::testing::opcode_matchers;
using Pass = CollectivePermuteDecomposer;

struct Decomposed {
  std::string cp_name;
  HloInstruction* after_all;
  HloInstruction* send;
  HloInstruction* recv;
  HloInstruction* send_done;
  HloInstruction* recv_done;
};

class DecomposerTest : public HloHardwareIndependentTestBase {
 protected:
  void AssertNoTranform(absl::string_view hlo, int64_t threshold = 0) {
    TF_ASSERT_OK(RunAndCheckHloRewrite(hlo, Pass(threshold), false));
  };
  auto Transform(absl::string_view hlo, int64_t threshold = 0) {
    return RunAndCheckHloRewrite(hlo, Pass(threshold), true);
  };
  void AssertTransform(absl::string_view hlo, int64_t threshold = 0) {
    TF_ASSERT_OK(RunAndCheckHloRewrite(hlo, Pass(threshold), true));
  };
  Decomposed FindComponents(HloModule* module, absl::string_view cp_name) {
    Decomposed result;
    result.cp_name = cp_name;
    result.after_all =
        FindInstruction(module, absl::StrCat(cp_name, "-after-all"));
    result.send = FindInstruction(module, absl::StrCat(cp_name, "-send"));
    result.recv = FindInstruction(module, absl::StrCat(cp_name, "-recv"));
    result.send_done =
        FindInstruction(module, absl::StrCat(cp_name, "-send-done"));
    result.recv_done =
        FindInstruction(module, absl::StrCat(cp_name, "-recv-done"));
    CHECK(result.after_all != nullptr) << cp_name;
    CHECK(result.send != nullptr) << cp_name;
    CHECK(result.recv != nullptr) << cp_name;
    CHECK(result.send_done != nullptr) << cp_name;
    CHECK(result.recv_done != nullptr) << cp_name;
    return result;
  }
};

const char* kSimpleHloWhileLoopTemplate = R"(
  HloModule module
  cond {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    n = u32[] constant(2)
    ROOT result = pred[] compare(i, n), direction=LT
  }

  $hlo_while_body

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c42 = f32[] constant(42.0)
    init = f32[64] broadcast(c42), dimensions={}
    while_init = (u32[], f32[64]) tuple(c0, init)
    while_result = (u32[], f32[64]) while(while_init), body=body, condition=cond
    ROOT result = f32[64] get-tuple-element(while_result), index=1
  }
)";

static std::string GetSimpleHloWhileLoopStr(absl::string_view hlo_while_body) {
  return absl::StrReplaceAll(kSimpleHloWhileLoopTemplate,
                             {
                                 {"$hlo_while_body", hlo_while_body},
                             });
}

TEST_F(DecomposerTest, WithCycleNotTransformed) {
  AssertNoTranform(GetSimpleHloWhileLoopStr(R"(
  body {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    data = get-tuple-element(param), index=1
    cp = f32[64] collective-permute(data), channel_id=1,
        source_target_pairs={{0,1}, {1,0}}
    ROOT result = tuple(i, cp)
  }
  )"));
}

TEST_F(DecomposerTest, ThresholdNotTransformed) {
  const int64_t kThreshold = 64 * 8;
  AssertNoTranform(GetSimpleHloWhileLoopStr(R"(
  body {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    data = get-tuple-element(param), index=1
    cp = f32[64] collective-permute(data),
        source_target_pairs={{0,1}, {1,2}, {2,3}}
    ROOT result = tuple(i, cp)
  }
  )"),
                   kThreshold);
}

TEST_F(DecomposerTest, Basic) {
  AssertTransform(GetSimpleHloWhileLoopStr(R"(
  body {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    data = get-tuple-element(param), index=1
    cp = f32[64] collective-permute(data), channel_id=1,
        source_target_pairs={{0,1}, {1,2}}
    ROOT result = tuple(i, cp)
  }
  )"));
}

TEST_F(DecomposerTest, NoChannelId) {
  AssertTransform(GetSimpleHloWhileLoopStr(R"(
  body {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    data = get-tuple-element(param), index=1
    cp = f32[64] collective-permute(data), source_target_pairs={{0,1}, {1,2}}
    ROOT result = tuple(i, cp)
  }
  )"));
}

TEST_F(DecomposerTest, OutsideOfWhileLoop) {
  AssertNoTranform(R"(HloModule test
    ENTRY test_computation {
      data = u32[] parameter(0)
      ROOT cp = u32[] collective-permute(data), channel_id=1,
          source_target_pairs={{0,1}, {1,2}}
    })");
}

TEST_F(DecomposerTest, ControlDependency_IndependentCPs) {
  absl::string_view hlo = R"(
    HloModule test

    cond {
      param = (u32[], f32[64], f32[64], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      n = u32[] constant(2)
      ROOT result = pred[] compare(i, n), direction=LT
    }

    body {
      param = (u32[], f32[64], f32[64], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data1 = f32[64] get-tuple-element(param), index=1
      data2 = f32[64] get-tuple-element(param), index=2
      cp3 = f32[64] collective-permute(data2), source_target_pairs={{6,7}}
      cp1 = f32[64] collective-permute(data1), source_target_pairs={{3,0}}
      cp2 = f32[64] collective-permute(data2),
          source_target_pairs={{0,1},{1,2},{2,3}}
      ROOT out = (u32[], f32[64], f32[64], f32[64]) tuple(i, cp2, cp3, cp1)
    }

    ENTRY test_computation {
      c0 = u32[] constant(0)
      c42 = f32[] constant(42.0)
      init = f32[64] broadcast(c42), dimensions={}
      while_init = (u32[], f32[64], f32[64], f32[64])
          tuple(c0, init, init, init)
      ROOT while_result = (u32[], f32[64], f32[64], f32[64]) while(while_init),
          body=body, condition=cond
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, Transform(hlo));
  Decomposed cp1 = FindComponents(module.get(), "cp1");
  Decomposed cp2 = FindComponents(module.get(), "cp2");
  Decomposed cp3 = FindComponents(module.get(), "cp3");
  // Sequence in tuple determines the port order and therefore control
  // dependency of consecutive CPs.
  EXPECT_THAT(cp3.recv->control_predecessors(), ElementsAre(cp2.send));
  EXPECT_THAT(cp1.recv->control_predecessors(), ElementsAre(cp3.send));
}

TEST_F(DecomposerTest, ControlDependency_BasicDependency) {
  const std::string kHlo = GetSimpleHloWhileLoopStr(R"(
    body {
      param = (u32[], f32[64]) parameter(0)
      i = get-tuple-element(param), index=0
      data = get-tuple-element(param), index=1
      cp-a = f32[64] collective-permute(data),
          source_target_pairs={{0,1}, {1,2}, {2,3}}
      cp-b = f32[64] collective-permute(cp-a), source_target_pairs={{3,0}}
      ROOT result = (u32[], f32[64]) tuple(i, cp-b)
    }
  )");
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, Transform(kHlo));
  Decomposed cp_a = FindComponents(module.get(), "cp-a");
  EXPECT_NE(FindInstruction(module.get(), "cp-b"), nullptr);
}

TEST_F(DecomposerTest, ControlDependency_MoreDependencies) {
  std::string kHlo = GetSimpleHloWhileLoopStr(R"(
    body {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[64] get-tuple-element(param), index=1

      // misordered names to assure that dependencies are honored
      cp1 = f32[64] collective-permute(data), source_target_pairs={{3,0}}
      cp2 = f32[64] collective-permute(cp1),
          source_target_pairs={{0,1},{1,2},{2,3}}
      cp3 = f32[64] collective-permute(cp2), source_target_pairs={{6,7}}
      ROOT out = (u32[], f32[64]) tuple(i, cp3)
    })");
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, Transform(kHlo));
  Decomposed cp1 = FindComponents(module.get(), "cp1");
  EXPECT_NE(FindInstruction(module.get(), "cp2"), nullptr);
  EXPECT_NE(FindInstruction(module.get(), "cp3"), nullptr);
}

void EnsurePreservedInfo(const HloInstruction* instr) {
  SCOPED_TRACE("AssurePreservedInfo for: " + instr->ToString());
  EXPECT_EQ(instr->channel_id().value(), 1);
  EXPECT_EQ(instr->metadata().op_name(), "op1/op2/add");
  EXPECT_EQ(instr->metadata().source_file(), "foo/bar/mysource.py");
  EXPECT_EQ(instr->metadata().source_line(), 35);
  EXPECT_THAT(
      instr->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}"));
}

std::string PipelineAttr(const HloInstruction* instr) {
  const FrontendAttributes& attr = instr->frontend_attributes();
  if (auto it = attr.map().find(kSendRecvPipelineAttr);
      it != attr.map().end()) {
    return it->second;
  }
  return "";
}
std::string OtherAttr(const HloInstruction* instr) {
  const FrontendAttributes& attributes = instr->frontend_attributes();
  return attributes.map().find("_xla_other_attribute")->second;
}

void EnsurePipelineAttr(Decomposed cp, std::string val) {
  SCOPED_TRACE("ExpectePipelineAttr for " + cp.cp_name);
  EXPECT_EQ(PipelineAttr(cp.recv), val);
  EXPECT_EQ(PipelineAttr(cp.send), val);
  EXPECT_EQ(PipelineAttr(cp.recv_done), val);
  EXPECT_EQ(PipelineAttr(cp.send_done), val);
}

void EnsureControlDependency(Decomposed cp) {
  SCOPED_TRACE("ExpectOpControlDependency for " + cp.cp_name);
  EXPECT_EQ(cp.recv->operand(0), cp.after_all);
  EXPECT_EQ(cp.send->operand(1), cp.after_all);
  EXPECT_EQ(cp.recv_done->operand(0), cp.recv);
  EXPECT_EQ(cp.send_done->operand(0), cp.send);

  EXPECT_THAT(cp.send->control_predecessors(), ElementsAre(cp.recv))
      << "Send should depend on recv when decoposed";
  EXPECT_THAT(cp.recv_done->control_predecessors(), ElementsAre(cp.send))
      << "Recv-done should depend on send when decoposed";
}

TEST_F(DecomposerTest, StructureAndMetadata) {
  std::string kHlo = GetSimpleHloWhileLoopStr(R"(
    body {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[64] get-tuple-element(param), index=1
      cp = f32[64] collective-permute(data), channel_id=1,
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}},
        metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py"
        source_line=35}
      ROOT result = (u32[], f32[64]) tuple(i, cp)
    })");
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, Transform(kHlo));
  Decomposed cp = FindComponents(module.get(), "cp");
  EnsurePreservedInfo(cp.send);
  EnsurePreservedInfo(cp.recv);
  EnsureControlDependency(cp);
}

TEST_F(DecomposerTest, Pipeline1) {
  absl::string_view hlo = R"(
    HloModule module

    cond {
      param = (u32[], u32[2]) parameter(0)
      count = get-tuple-element(param), index=0
      ub = u32[] constant(2)
      ROOT result = pred[] compare(count, ub), direction=LT
    }

    body {
      param = (u32[], u32[2]) parameter(0)
      count = get-tuple-element(param), index=0
      send-data = get-tuple-element(param), index=1

      cp = u32[2] collective-permute(send-data), channel_id=1,
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}},
        frontend_attributes={_xla_other_attribute="xyz"}

      c1 = u32[] constant(1)
      new_count = u32[] add(count, c1)

      r = u32[2] broadcast(c1), dimensions={}
      s = u32[2] add(r, cp)

      ROOT result = (u32[], u32[2]) tuple(new_count, s)
    }

    ENTRY test_computation {
      c0 = u32[] constant(0)
      c1 = u32[] constant(1)
      r = u32[] replica-id()
      a = u32[] add(c1, r)
      init = u32[2] broadcast(a), dimensions={}
      while_init = (u32[], u32[2]) tuple(c0, init)
      while_result = (u32[], u32[2]) while(while_init), body=body,
          condition=cond
      ROOT result = u32[2] get-tuple-element(while_result), index=1
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, Transform(hlo));
  Decomposed cp = FindComponents(module.get(), "cp");
  EnsurePipelineAttr(cp, "0");
  EXPECT_EQ(OtherAttr(cp.recv), "xyz") << "Preseving other attributes";
  EXPECT_EQ(OtherAttr(cp.send), "xyz") << "Preseving other attributes";
  EnsureControlDependency(cp);
}

TEST_F(DecomposerTest, ForwardPipeline2) {
  absl::string_view hlo = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    cp_fwd = u32[2] collective-permute(send-data), channel_id=2,
      source_target_pairs={{0,1}, {1,2}, {2,3}}

    cp_back = u32[2] collective-permute(send-data), channel_id=1,
      source_target_pairs={{3,0}}

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[2] broadcast(compare0), dimensions={}
    recv-data = u32[2] select(compare, cp_back, cp_fwd)

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    ROOT result = (u32[], u32[2]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}
    while_init = (u32[], u32[2]) tuple(c0, init)
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond
    ROOT result = u32[2] get-tuple-element(while_result), index=1
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, Transform(hlo));

  Decomposed cp_back = FindComponents(module.get(), "cp_back");
  Decomposed cp_fwd = FindComponents(module.get(), "cp_fwd");

  EXPECT_EQ(cp_back.recv->channel_id().value(), 1);
  EXPECT_EQ(cp_fwd.recv->channel_id().value(), 2);
  EnsurePipelineAttr(cp_back, "0");
  EnsurePipelineAttr(cp_fwd, "1");
  EnsureControlDependency(cp_back);
  EnsureControlDependency(cp_fwd);
  EXPECT_THAT(cp_fwd.recv->control_predecessors(), ElementsAre(cp_back.send))
      << "Per sequence of select operands, cp_back should come before cp_fwd";
}

TEST_F(DecomposerTest, ForwardPipelineWithMatmul) {
  // The HLO module below is generated by passing the HLO in
  // CollectiveOpsTest.CollectivePermute_CircularPipelinePreOptimization through
  // the collective_permute_cycle_decomposer.transformation.
  absl::string_view hlo = R"(
  HloModule test

  while_body {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    iter_increment = u32[] constant(1)
    next_iter = u32[] add(iter, iter_increment)
    partition-id = u32[] partition-id()
    zero = u32[] constant(0)
    compare = pred[] compare(partition-id, zero), direction=EQ
    broadcast = pred[2,2] broadcast(compare), dimensions={}

    weights = f32[2,2] get-tuple-element(inputs), index=2
    data = f32[2,2] get-tuple-element(inputs), index=1

    cp_back = f32[2,2] collective-permute(data), channel_id=1,
      source_target_pairs={{3,0}},
      frontend_attributes={_xla_send_recv_validation="{{3,10}}"}
    cp_fwd = f32[2,2] collective-permute(data), channel_id=2,
      source_target_pairs={{0,1},{1,2},{2,3}},
      frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9}}"}

    select = f32[2,2] select(broadcast, cp_back, cp_fwd)

    matmul = f32[2,2] dot(weights, select), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, matmul, weights)
  }

  while_cond {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    max_iter = u32[] constant(3)
    ROOT compare = pred[] compare(iter, max_iter), direction=LT
  }

  ENTRY test_computation {
    start_iter = u32[] constant(0)
    input_data = f32[2,2] parameter(0)
    input_weights = f32[2,2] parameter(1)
    input = (u32[], f32[2,2], f32[2,2]) tuple(start_iter, input_data,
        input_weights)
    while_result = (u32[], f32[2,2], f32[2,2]) while(input),
        condition=while_cond, body=while_body
    ROOT data_out = f32[2,2] get-tuple-element(while_result), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, Transform(hlo));
  Decomposed cp_back = FindComponents(module.get(), "cp_back");
  Decomposed cp_fwd = FindComponents(module.get(), "cp_fwd");
  EXPECT_EQ(cp_back.recv->channel_id().value(), 1);
  EXPECT_EQ(cp_fwd.recv->channel_id().value(), 2);
  EnsurePipelineAttr(cp_back, "0");
  EnsurePipelineAttr(cp_fwd, "1");
  EnsureControlDependency(cp_back);
  EnsureControlDependency(cp_fwd);
  EXPECT_THAT(cp_fwd.recv->control_predecessors(), ElementsAre(cp_back.send));
}

TEST_F(DecomposerTest, BackwardPipeline2) {
  absl::string_view hlo = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    cp_fwd = u32[2] collective-permute(send-data), channel_id=1,
      source_target_pairs={{1,0},{2,1},{3,2}}

    cp_back = u32[2] collective-permute(send-data), channel_id=2,
      source_target_pairs={{0,3}}

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=NE
    compare = pred[2] broadcast(compare0), dimensions={}
    recv-data = u32[2] select(compare, cp_fwd, cp_back)

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    ROOT result = (u32[], u32[2]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}
    while_init = (u32[], u32[2]) tuple(c0, init)
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond
    ROOT result = u32[2] get-tuple-element(while_result), index=1
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, Transform(hlo));
  Decomposed cp_back = FindComponents(module.get(), "cp_back");
  Decomposed cp_fwd = FindComponents(module.get(), "cp_fwd");
  EXPECT_EQ(cp_back.recv->channel_id().value(), 2);
  EXPECT_EQ(cp_fwd.recv->channel_id().value(), 1);

  EnsurePipelineAttr(cp_back, "0");
  EnsurePipelineAttr(cp_fwd, "1");
  EnsureControlDependency(cp_back);
  EnsureControlDependency(cp_fwd);
  EXPECT_THAT(cp_back.recv->control_predecessors(), ElementsAre(cp_fwd.send))
      << "Per sequence of select operands, cp_fwd should come before cp_back";
}

}  // namespace
}  // namespace xla
