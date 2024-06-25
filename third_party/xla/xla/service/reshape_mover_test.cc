/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/reshape_mover.h"

#include <memory>
#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace m = xla::match;

class ReshapeMoverTest : public HloTestBase {
 protected:
  // ReshapeMover relies on algsimp for cleanup.
  absl::Status RunPass(HloModule* module, bool change_expected,
                       ReshapeMoverOptions options = ReshapeMoverOptions{}) {
    TF_ASSIGN_OR_RETURN(bool changed,
                        RunHloPass(ReshapeMover(options), module));
    SCOPED_TRACE(module->ToString());
    EXPECT_EQ(changed, change_expected);
    TF_EXPECT_OK(RunHloPass(HloVerifier(HloVerifierOpts()), module).status());
    TF_EXPECT_OK(RunHloPass(HloPassFix<AlgebraicSimplifier>(
                                AlgebraicSimplifierOptions()),
                            module)
                     .status());
    return absl::OkStatus();
  }
};

TEST_F(ReshapeMoverTest, ReshapesWithDifferentInputShapesNotMoved) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      reshape0 = f32[8,7] reshape(f32[1,8,1,7] parameter(0))
      reshape1 = f32[8,7] reshape(f32[1,8,7,1] parameter(1))
      ROOT add = add(reshape0, reshape1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/false));
}

TEST_F(ReshapeMoverTest, OneConstantAndOneReshapesOnRngNotMoved) {
  // The reshape should not be moved, since rng0 is trivially reshapable and
  // therefore there are no nontrivial reshapes to move.
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      rng = f32[1,8,1,7,1] rng(f32[] constant(0), f32[] constant(1)), distribution=rng_uniform
      ROOT add = add(f32[8,7] reshape(rng), f32[8,7] constant({...}))
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/false));
}

TEST_F(ReshapeMoverTest, EquivalentReshapesMoved) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      reshape0 = f32[8,7] reshape(f32[1,8,1,7] parameter(0))
      reshape1 = f32[8,7] reshape(f32[1,8,1,7] parameter(1))
      ROOT add = f32[8,7] add(reshape0, reshape1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(ReshapeMoverTest, SinkReshapeBelowSelect) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      ROOT select = f32[2,3] select(
        pred[2,3] reshape(pred[6] parameter(0)),
        f32[2,3] reshape(f32[6] parameter(1)),
        f32[2,3] reshape(f32[6] parameter(2)))
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true));
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Select(m::Parameter(0), m::Parameter(1),
                                              m::Parameter(2)))));
}

TEST_F(ReshapeMoverTest, SinkReshapeBelowSelectWithConstant) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      ROOT select = f32[2,3] select(
        pred[2,3] reshape(pred[6] parameter(0)),
        f32[2,3] reshape(f32[6] parameter(1)),
        f32[2,3] constant({...}))
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true));
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Select(m::Parameter(0), m::Parameter(1),
                                              m::Reshape(m::Constant())))));
}

TEST_F(ReshapeMoverTest, OneParameterAndOneReshapeNotMoved) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      reshape0 = f32[8,7] reshape(f32[1,8,1,7] parameter(0))
      ROOT add = add(reshape0, f32[8,7] parameter(1))
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/false));
}

TEST_F(ReshapeMoverTest, DontSinkReshapesOfConstants) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      ROOT select = select(
        pred[3,2] parameter(0),
        f32[3,2] reshape(f32[2,3] constant({...})),
        f32[3,2] reshape(f32[2,3] constant({...})))
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/false));
}

TEST_F(ReshapeMoverTest, OneNontrivialReshapeMoved) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      ROOT add = add(
        f32[3,2] reshape(f32[2,3] parameter(0)),
        f32[3,2] constant({...}))
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true));
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(
                  m::Add(m::Parameter(0), m::Reshape(m::Constant())))));
}

TEST_F(ReshapeMoverTest, MultipleReshapes) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      add0 = f32[8,7,1] add(
        f32[8,7,1] reshape(f32[1,8,1,7] parameter(0)),
        f32[8,7,1] reshape(f32[1,8,1,7] parameter(1)))
      ROOT add1 = f32[8,7] add(
        f32[8,7] reshape(add0),
        f32[8,7] reshape(f32[8,7,1] parameter(2)))
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true));
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(
                  m::Add(m::Reshape(m::Add(m::Parameter(0), m::Parameter(1))),
                         m::Parameter(2)))));
}

TEST_F(ReshapeMoverTest, SinkTransposeAcrossBroadcastScalar) {
  const std::string hlo_string = R"(
    HloModule TransposeMulInversedTransposeModule
    ENTRY TransposeMulInversedTranspose {
      src0 = f32[20,8]{1,0} parameter(0)
      transpose0 = f32[8,20]{1,0} transpose(src0), dimensions={1,0}
      src1 = f32[] parameter(1)
      broadcast0 = f32[8,20]{1,0} broadcast(src1), dimensions={}
      ROOT multiply0 = f32[8,20]{1,0} multiply(transpose0, broadcast0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true));
  SCOPED_TRACE(m->ToString());

  // ReshapeMover transforms to transpose(broadcast(param(1))), and then algsimp
  // transforms to broadcast'(param(1)).
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Transpose(m::Multiply(
                  m::Parameter(0), m::Broadcast(m::Parameter(1))))));
}

TEST_F(ReshapeMoverTest, ReshapeWithUsersOutsideCandidatesNotSink) {
  const std::string hlo_string = R"(
    HloModule ReshapeWithUsersOutsideCandidates
    ENTRY ReshapeWithMultipleUsers {
      param0 = f32[20,8]{1,0} parameter(0)
      reshape0 = f32[8,20]{1,0} reshape(param0)
      param1 = f32[] parameter(1)
      broadcast0 = f32[8,20]{1,0} broadcast(param1), dimensions={}
      param2 = f32[20,8]{1,0} parameter(2)
      reshape1 = f32[8,20]{1,0} reshape(param2)
      param3 = f32[20,8]{1,0} parameter(3)
      reshape2 = f32[8,20]{1,0} reshape(param3)
      param4 = f32[8,20]{1,0} parameter(4)
      add0 = f32[8,20]{1,0} add(reshape0, broadcast0)
      add1 = f32[8,20]{1,0} add(reshape0, reshape1)
      add2 = f32[8,20]{1,0} add(reshape1, param4)
      ROOT tuple = (f32[8,20]{1,0},f32[8,20]{1,0},
        f32[8,20]{1,0}) tuple(add0, add1, add2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/false));
}

TEST_F(ReshapeMoverTest, ReshapeNoUsersOutsideCandidatesSink1) {
  const std::string hlo_string = R"(
    HloModule ReshapeNoUsersOutsideCandidates1
    ENTRY ReshapeWithMultipleUsers1 {
      param0 = f32[20,8]{1,0} parameter(0)
      reshape0 = f32[8,20]{1,0} reshape(param0)
      param1 = f32[] parameter(1)
      broadcast0 = f32[8,20]{1,0} broadcast(param1), dimensions={}
      param2 = f32[20,8]{1,0} parameter(2)
      reshape1 = f32[8,20]{1,0} reshape(param2)
      param3 = f32[20,8]{1,0} parameter(3)
      reshape2 = f32[8,20]{1,0} reshape(param3)
      add0 = f32[8,20]{1,0} add(reshape0, broadcast0)
      add1 = f32[8,20]{1,0} add(reshape0, reshape1)
      add2 = f32[8,20]{1,0} add(reshape1, reshape2)
      ROOT tuple = (f32[8,20]{1,0},f32[8,20]{1,0},
        f32[8,20]{1,0}) tuple(add0, add1, add2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true));
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Reshape(m::Add(m::Parameter(0), m::Broadcast(m::Parameter(1)))),
          m::Reshape(m::Add(m::Parameter(0), m::Parameter(2))),
          m::Reshape(m::Add(m::Parameter(2), m::Parameter(3))))));
}

TEST_F(ReshapeMoverTest, ReshapeNoUsersOutsideCandidatesSink2) {
  const std::string hlo_string = R"(
    HloModule ReshapeNoUsersOutsideCandidates2
    ENTRY ReshapeWithMultipleUsers2 {
      param0 = f32[20,8]{1,0} parameter(0)
      reshape0 = f32[8,20]{1,0} reshape(param0)
      ROOT add0 = f32[8,20]{1,0} add(reshape0, reshape0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true));
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Add())));
}

TEST_F(ReshapeMoverTest, ReshapeOfRank1BroadcastIsNotTrivial) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      a = f32[2,3] broadcast(f32[2] parameter(0)), dimensions={0}
      b = f32[2,3] reshape(f32[6] parameter(1))
      ROOT add0 = add(a, b)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/false));
}

TEST_F(ReshapeMoverTest, ReshapeOfRank1BroadcastIsTrivial) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      a = f32[2,3] broadcast(f32[2] parameter(0)), dimensions={0}
      b = f32[2,3] reshape(f32[6] parameter(1))
      ROOT add0 = add(a, b)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));

  ReshapeMoverOptions options;
  options.reshape_of_1d_broadcast_is_cheap = true;
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true, options));

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Reshape(
          m::Add(m::Reshape(m::Broadcast(m::Parameter(0))), m::Parameter(1)))));
}

TEST_F(ReshapeMoverTest, ReshapeOfRank2BroadcastIsAllowed) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      a = f32[2,3,35] broadcast(f32[2,3] parameter(0)), dimensions={0,1}
      b = f32[2,3,35] reshape(f32[2,3,5,7] parameter(1))
      ROOT add0 = add(a, b)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  ReshapeMoverOptions options;
  options.reshape_of_1d_broadcast_is_cheap = true;
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true, options));
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(
                  m::Add(m::Broadcast(m::Parameter(0)), m::Parameter(1)))));
}

TEST_F(ReshapeMoverTest, SinkDisallowedIfReshapeChangesBroadcastDims) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      a = f32[2,3,35] broadcast(f32[2,3] parameter(0)), dimensions={0,1}
      b = f32[2,3,35] reshape(f32[6,5,7] parameter(1))
      ROOT add0 = add(a, b)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/false));
}

TEST_F(ReshapeMoverTest, TransposeOfBroadcastIsAllowed) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      a = f32[2,3] broadcast(f32[2] parameter(0)), dimensions={0}
      b = f32[2,3] transpose(f32[3,2] parameter(1)), dimensions={1,0}

      ROOT add0 = add(a, b)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true));
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Transpose(
                  m::Add(m::Broadcast(m::Parameter(0)), m::Parameter(1)))));
}

TEST_F(ReshapeMoverTest, TransposeReordersBroadcastDims) {
  const std::string hlo_string = R"(
    HloModule test
    ENTRY test {
      a = f32[2,3,5] broadcast(f32[2,3] parameter(0)), dimensions={0,1}
      b = f32[2,3,5] transpose(f32[3,2,5] parameter(1)), dimensions={1,0,2}

      ROOT add0 = add(a, b)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/false));
}

TEST_F(ReshapeMoverTest, ShardingConsistencyPreservation) {
  const std::string hlo_string = R"(
    HloModule module

    ENTRY entry {
      copy.2424 = bf16[3,16,128]{2,1,0} parameter(0), sharding={replicated}
      dot.987 = bf16[3,16,128,4096]{3,2,1,0} parameter(1), sharding={devices=[1,8,1,1]0,1,2,3,4,5,6,7}
      reshape.5843 = bf16[3,16,128,1,4096]{4,3,2,1,0} reshape(dot.987), sharding={devices=[1,8,1,1,1]0,1,2,3,4,5,6,7}
      transpose.21172 = bf16[3,1,4096,16,128]{2,1,4,3,0} transpose(reshape.5843), dimensions={0,3,4,1,2}, sharding={devices=[1,1,1,8,1]0,1,2,3,4,5,6,7}
      reshape.291 = bf16[3,16,128]{2,1,0} reshape(copy.2424), sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
      broadcast.21176 = bf16[3,1,4096,16,128]{4,3,2,1,0} broadcast(reshape.291), dimensions={0,3,4}, sharding={devices=[1,1,1,8,1]0,1,2,3,4,5,6,7}
      multiply.21177 = bf16[3,1,4096,16,128]{2,1,4,3,0} multiply(transpose.21172, broadcast.21176), sharding={devices=[1,1,1,8,1]0,1,2,3,4,5,6,7}
      ROOT slice.21180 = bf16[1,1,4096,16,128]{4,3,2,1,0} slice(multiply.21177), slice={[1:2], [0:1], [0:4096], [0:16], [0:128]}, sharding={devices=[1,1,1,8,1]0,1,2,3,4,5,6,7}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(m.get(), /*change_expected=*/true));
  auto elementwise_op = FindInstruction(m.get(), HloOpcode::kMultiply);
  EXPECT_FALSE(elementwise_op->has_sharding());
}

}  // namespace
}  // namespace xla
