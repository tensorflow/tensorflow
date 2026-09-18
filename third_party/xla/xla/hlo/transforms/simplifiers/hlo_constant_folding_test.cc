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

#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
namespace m = xla::match;
using HloConstantFoldingTest = HloHardwareIndependentTestBase;

TEST_F(HloConstantFoldingTest, ConvertF32ToS64) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(S64, {}), input));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert().WithOperand(0, m::Op().Is(input))));

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Constant()));
  EXPECT_EQ(
      computation->root_instruction()->literal().GetFirstElement<int64_t>(),
      42);
}

TEST_F(HloConstantFoldingTest, ConvertS64ToF32) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(42)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(F32, {}), input));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert().WithOperand(0, m::Op().Is(input))));

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Constant()));
  EXPECT_EQ(computation->root_instruction()->literal().GetFirstElement<float>(),
            42.0f);
}

TEST_F(HloConstantFoldingTest, ConvertF32ArrayToS64Array) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({42.0f, 19.0f})));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(S64, {2}), input));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert().WithOperand(0, m::Op().Is(input))));

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Constant()));
  EXPECT_EQ(computation->root_instruction()->literal().Get<int64_t>({0}), 42);
  EXPECT_EQ(computation->root_instruction()->literal().Get<int64_t>({1}), 19);
}

TEST_F(HloConstantFoldingTest, Concatenate) {
  const struct TestConfig {
    int concat_dimension;
    std::vector<int64_t> dimensions;
    std::vector<int64_t> concat_sizes;
  } test_configs[] = {
      {1, {11, 0, 7, 5, 9}, {2, 5, 7, 11}},
      {3, {1, 4, 17, 0, 8}, {1, 3, 9, 12}},
  };

  for (auto& test_config : test_configs) {
    HloComputation::Builder builder(TestName());
    std::vector<int64_t> dimensions(test_config.dimensions.begin(),
                                    test_config.dimensions.end());
    int64_t concat_size = 0;
    std::vector<HloInstruction*> operands;
    for (auto csize : test_config.concat_sizes) {
      dimensions[test_config.concat_dimension] = csize;
      concat_size += csize;
      auto literal = LiteralUtil::CreateFromDimensions(F32, dimensions);
      HloInstruction* insn = builder.AddInstruction(
          HloInstruction::CreateConstant(std::move(literal)));
      operands.push_back(insn);
    }
    dimensions[test_config.concat_dimension] = concat_size;
    Shape shape = ShapeUtil::MakeShape(F32, dimensions);
    builder.AddInstruction(HloInstruction::CreateConcatenate(
        shape, operands, test_config.concat_dimension));
    auto module = CreateNewVerifiedModule();
    auto computation = module->AddEntryComputation(builder.Build());

    HloConstantFolding const_folder;
    TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
    EXPECT_TRUE(result);

    HloInstruction* root = computation->root_instruction();
    EXPECT_THAT(root, GmockMatch(m::Constant()));
    EXPECT_TRUE(ShapeUtil::Equal(root->shape(), shape));
  }
}

TEST_F(HloConstantFoldingTest, Slice) {
  HloComputation::Builder builder(TestName());
  const int64_t dimensions[] = {11, 8, 7, 5, 9};
  const int64_t slice_start[] = {4, 2, 3, 1, 5};
  const int64_t slice_limits[] = {10, 8, 6, 5, 9};
  const int64_t slice_strides[] = {1, 1, 1, 1, 1};
  TF_ASSERT_OK_AND_ASSIGN(auto literal,
                          LiteralUtil::CreateRandomLiteral<F32>(
                              ShapeUtil::MakeShape(F32, dimensions), 0.0, 1.0));
  HloInstruction* literal_instruction = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
  Shape shape = ShapeUtil::MakeShape(F32, {6, 6, 3, 4, 4});
  builder.AddInstruction(HloInstruction::CreateSlice(
      shape, literal_instruction, slice_start, slice_limits, slice_strides));
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), shape));
}

TEST_F(HloConstantFoldingTest, TransposeConstantFold) {
  HloComputation::Builder builder(TestName());
  const int64_t dimensions[] = {11, 8, 7, 5, 9};
  TF_ASSERT_OK_AND_ASSIGN(auto literal,
                          LiteralUtil::CreateRandomLiteral<F32>(
                              ShapeUtil::MakeShape(F32, dimensions), 0.0, 1.0));
  auto literal_clone = literal.Clone();
  HloInstruction* literal_instruction = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
  Shape shape = ShapeUtil::MakeShape(F32, {8, 7, 11, 9, 5});
  const int64_t permutation[] = {1, 2, 0, 4, 3};
  builder.AddInstruction(
      HloInstruction::CreateTranspose(shape, literal_instruction, permutation));
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), shape));

  using NativeT = typename primitive_util::PrimitiveTypeToNative<F32>::type;
  bool matched = true;
  root->literal().EachCell<NativeT>(
      [&](absl::Span<const int64_t> indices, NativeT value) {
        std::vector<int64_t> rindexes = PermuteInverse(indices, permutation);
        matched = matched && (value == literal_clone.Get<NativeT>(rindexes));
      });
  EXPECT_TRUE(matched);
}

const char* const kConstantFoldReduce = R"(
  HloModule ConstantFoldReduce

  add {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT add = s32[] add(a, b)
  }

  ENTRY r {
    x = s32[3] constant({1, 2, 3})
    init = s32[] constant(0)
    ROOT reduce = s32[] reduce(x, init), dimensions={0}, to_apply=add
  })";

TEST_F(HloConstantFoldingTest, ConstantFoldReduce) {
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnVerifiedModule(kConstantFoldReduce));
  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(m.get()));
  EXPECT_TRUE(result);

  EXPECT_EQ(6, m->entry_computation()
                   ->root_instruction()
                   ->literal()
                   .GetFirstElement<int32_t>());
}

constexpr absl::string_view kConstantFoldReduceWithMetadata = R"(
  HloModule ConstantFoldReduce

  add {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT add = s32[] add(a, b)
  }

  ENTRY r {
    x = s32[3] constant({1, 2, 3}), metadata={op_name="constant"}
    init = s32[] constant(0), metadata={op_name="zero_constant"}
    ROOT reduce = s32[] reduce(x, init), metadata={op_name="reduce"}, dimensions={0}, to_apply=add
  })";

TEST_F(HloConstantFoldingTest, ConstantFoldReduceCheckMetadata) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto m, ParseAndReturnVerifiedModule(kConstantFoldReduceWithMetadata));
  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(m.get()));
  EXPECT_TRUE(result);
  OpMetadata reduce_metadata;
  reduce_metadata.set_op_name("reduce");
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              AllOf(op::Constant(), op::Metadata(reduce_metadata)));
}

TEST_F(HloConstantFoldingTest, ConstantFoldReduceNoLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnVerifiedModule(kConstantFoldReduce));
  HloInstruction* add = (*m->computations().begin())->root_instruction();
  LayoutUtil::ClearLayout(add->mutable_shape());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(m.get()));
  EXPECT_TRUE(result);

  EXPECT_EQ(6, m->entry_computation()
                   ->root_instruction()
                   ->literal()
                   .GetFirstElement<int32_t>());
}

const char* const kConstantFoldLargePad = R"(
  HloModule ConstantFoldLargePad

  ENTRY r {
    a = f32[1,1,1] constant({{{7}}})
    b = f32[] constant(42)
    ROOT pad = f32[2048,2048,128] pad(a, b), padding=1024_1023x1024_1023x64_63
  })";

TEST_F(HloConstantFoldingTest, DoesNotFoldLargePad) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kConstantFoldLargePad));
  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_FALSE(result);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Pad(m::Constant(), m::Constant())));
}

TEST_F(HloConstantFoldingTest, DoesNotFoldPadBroadcast) {
  const char* const kConstantFoldPadBroadcast = R"(
  HloModule ConstantFoldLargePad

  ENTRY r {
    a = f32[] constant(239)
    broadcast_a = f32[4] broadcast(a), dimensions={}
    b = f32[] constant(42)
    ROOT pad = f32[8] pad(f32[4] broadcast_a, f32[] b), padding=4_0
  })";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kConstantFoldPadBroadcast));
  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_FALSE(result);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Pad(m::Broadcast(), m::Constant())));
}

TEST_F(HloConstantFoldingTest, DoesNotFoldSlicesWithLargeOperand) {
  const char* const kModuleStr = R"(
  HloModule test

  ENTRY r {
    a = f32[] constant(42)
    broadcast = f32[1000000000]{0} broadcast(a), dimensions={}
    slice1 = f32[10000]{0} slice(broadcast), slice={[0:10000]}
    slice2 = f32[10000]{0} slice(broadcast), slice={[10000:20000]}
    ROOT add = f32[10000]{0} add(slice1, slice2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_FALSE(result);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::Slice(), m::Slice())));
}

TEST_F(HloConstantFoldingTest, DontFoldSubcomputationContainingAfterAll) {
  const char* const kModuleStr = R"(
  HloModule test

  Fn {
    tok = token[] after-all()
    ROOT root = f32[10] iota(), iota_dimension=0
  }

  ENTRY entry {
    ROOT call = f32[10] call(), to_apply=Fn
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloConstantFoldingTest,
       DontFoldSubcomputationTransitivelyContainingRng) {
  const char* const kModuleStr = R"(
  HloModule test

  InnerFn {
    c0 = f32[] constant(0)
    c1 = f32[] constant(1)
    ROOT rng = f32[10] rng(c0, c1), distribution=rng_uniform
  }

  Fn {
    ROOT fusion = f32[10] fusion(), kind=kLoop, calls=InnerFn
  }

  ENTRY entry {
    ROOT call = f32[10] call(), to_apply=Fn
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloConstantFoldingTest, ConstantFoldCopyOp) {
  // Replace %copy.3 with %constant.2
  const char* const kModuleStr = R"(
  HloModule m
  ENTRY main {
    %p0 = f32[] parameter(0)
    %constant.2 = f32[] constant(0)
    ROOT %copy.3 = f32[] copy(f32[] %constant.2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Constant()));
}

TEST_F(HloConstantFoldingTest, DontFoldCopyOp_NonSafelyRemovableOp) {
  // copy.3 is not SafelyRemovable (has control-predecessors)
  // Skip ConstantFolding
  const char* const kModuleStr = R"(
  HloModule m
  ENTRY main {
    %p0 = f32[] parameter(0)
    %copy.1 = f32[] copy(f32[] %p0)
    %constant.2 = f32[] constant(0)
    ROOT %copy.3 = f32[] copy(f32[] %constant.2), control-predecessors={%copy.1}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloConstantFoldingTest, FoldOpsWhereOneOperandIsBroadcast) {
  const char* const kModuleStr = R"(
  HloModule test

  ENTRY entry {
    not_folded1 = f32[4] broadcast(f32[] constant(1))
    not_folded2 = add(f32[4] broadcast(f32[] constant(2)),
                      f32[4] broadcast(f32[] constant(3)))
    folded1 = add(f32[4] broadcast(f32[] constant(5)),
                  f32[4] constant({0,1,2,3}))
    folded2 = add(f32[4] constant({0,1,2,3}),
                  f32[4] broadcast(f32[] constant(5)))
    ROOT root = tuple(not_folded1, not_folded2, folded1, folded2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Broadcast(m::Constant()),
                                  m::Add(m::Broadcast(m::Constant()),
                                         m::Broadcast(m::Constant())),
                                  m::Constant(),
                                  m::Constant()  //
                                  )));
}

TEST_F(HloConstantFoldingTest, AgressiveFoldOpsWhereBothOperandAreBroadcast) {
  const char* const kModuleStr = R"(
  HloModule test

  ENTRY entry {
    not_folded1 = f32[4] broadcast(f32[] constant(1))
    folded1 = add(f32[4] broadcast(f32[] constant(2)),
                      f32[4] broadcast(f32[] constant(3)))
    folded2 = add(f32[4] broadcast(f32[] constant(5)),
                  f32[4] constant({0,1,2,3}))
    folded3 = add(f32[4] constant({0,1,2,3}),
                  f32[4] broadcast(f32[] constant(5)))
    ROOT root = tuple(not_folded1, folded1, folded2, folded3)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding(HloConstantFolding::Level::kAggressive);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Broadcast(m::Constant()),
                                  m::Constant(),  //
                                  m::Constant(),  //
                                  m::Constant()   //
                                  )));
}

TEST_F(HloConstantFoldingTest, FoldOpsWhereOneOperandIsIota) {
  const char* const kModuleStr = R"(
  HloModule test

  ENTRY entry {
    iota = f32[4] iota(), iota_dimension=0
    not_folded1 = add(f32[4] iota,
                      f32[4] iota)
    folded1 = add(f32[4] iota,
                  f32[4] constant({0,1,2,3}))
    folded2 = add(f32[4] constant({0,1,2,3}),
                  f32[4] iota)
    ROOT root = tuple(iota, not_folded1, folded1, folded2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Iota(),                     //
                                  m::Add(m::Iota(), m::Iota()),  //
                                  m::Constant(),                 //
                                  m::Constant())));
}

TEST_F(HloConstantFoldingTest, FoldInt4Ops) {
  const char* const kModuleStr = R"(
  HloModule test

  ENTRY entry {
    c0 = s4[2]{0:E(4)} constant({1, 2})
    c1 = s4[2]{0:E(4)} constant({3, 4})
    add1 = s4[2]{0:E(4)} add(c0, c1)
    c2 = s4[]{:E(4)} constant(5)
    add2 = s4[2]{0:E(4)} add(c0, s4[2]{0:E(4)} broadcast(c2))
    ROOT root = tuple(add1, add2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  auto is_4_bit = [](const HloInstruction* instr) {
    return instr->shape().layout().element_size_in_bits() == 4;
  };
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Constant().WithPredicate(is_4_bit),
                                  m::Constant().WithPredicate(is_4_bit))));
}

TEST_F(HloConstantFoldingTest, BigReduceWindow) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test

    add_bf16 {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT add = bf16[] add(lhs, rhs)
    }

    ENTRY accumulated_all_reduce {
      x = bf16[160,10,10,512]{3,2,1,0} broadcast(bf16[] constant(1.0))
      init = bf16[] constant(0)
      ROOT reduce-window = reduce-window(x, init), window={size=1x2x2x1 stride=1x2x2x1}, to_apply=add_bf16
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
}

TEST_F(HloConstantFoldingTest, TimingConsumingTest) {
  constexpr absl::string_view mod_str = R"(
    HloModule jit_f, entry_computation_layout={()->f32[]}
    region_0.4 {
      Arg_0.5 = f32[] parameter(0)
      Arg_1.6 = f32[] parameter(1)
      ROOT add.7 = f32[] add(Arg_0.5, Arg_1.6)
    }

    ENTRY main.9 {
      constant.1 = f32[] constant(1)
      broadcast.2 = f32[32,999,40,512]{3,2,1,0} broadcast(constant.1), dimensions={}
      constant.3 = f32[] constant(0)
      ROOT reduce.8 = f32[] reduce(broadcast.2, constant.3), dimensions={0,1,2,3}, to_apply=region_0.4
    }
   )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(mod_str));
  HloConstantFolding const_fold;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&const_fold, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloConstantFoldingTest, FoldWhile) {
  constexpr absl::string_view mod_str = R"(
    HloModule test
    condition_fn
    {
      parameter = (s32[], s32[10]) parameter(0)
      index = s32[] get-tuple-element(parameter), index=0
      ROOT compare.1 = pred[] compare(index, s32[] constant(5)), direction=LT
    }

    body_fn
    {
      parameter = (s32[], s32[10]) parameter(0)
      index = s32[] get-tuple-element(parameter), index=0
      value = s32[10] get-tuple-element(parameter), index=1
      incremented = s32[] add(index, s32[] constant(1))
      ROOT result = (s32[], s32[10]) tuple(incremented, value)
    }

    ENTRY main.9 {
      constant.1 = s32[] constant(0)
      broadcast.1 = s32[10] broadcast(s32[] constant(1))
      tuple_arg = (s32[], s32[10]) tuple(constant.1, broadcast.1)
      ROOT while = (s32[], s32[10]) while(tuple_arg), condition=condition_fn, body=body_fn
    }
   )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(mod_str));
  HloConstantFolding const_fold(HloConstantFolding::Level::kAggressive);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&const_fold, module.get()));
  EXPECT_TRUE(result);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Constant()));
}

TEST_F(HloConstantFoldingTest, FoldCall) {
  const char* const kModuleStr = R"(
    HloModule test

    Fn {
      param0 = f32[] parameter(0)
      param1 = f32[] parameter(1)
      ROOT add = f32[] add(param0, param1)
    }

    ENTRY entry {
      constant.0 = f32[] constant(1)
      constant.1 = f32[] constant(2)
      ROOT call = f32[] call(constant.0, constant.1), to_apply=Fn
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Constant()));
}

TEST_F(HloConstantFoldingTest, FoldCallToFft) {
  const char* const kModuleStr = R"(
    HloModule test

    Fn {
      param0 = c64[8] parameter(0)
      ROOT fft = c64[8] fft(param0), fft_type=FFT, fft_length={32}
    }

    ENTRY entry {
      constant.0 = c64[8] constant({(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)})
      ROOT call = c64[8] call(constant.0), to_apply=Fn
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloConstantFoldingTest, InterproceduralSingleCallsite) {
  const char* const kModuleStr = R"(
    HloModule test

    Fn {
      param0 = f32[8] parameter(0)
      param1 = f32[8] parameter(1)
      iota = f32[8] iota(), iota_dimension=0
      add.0 = f32[8] add(param0, iota)
      ROOT add.1 = f32[8] add(add.0, param1)
    }

    ENTRY entry {
      entry.param = f32[8] parameter(0)
      constant.0 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -1})
      ROOT call = f32[8] call(constant.0, entry.param), to_apply=Fn
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  HloComputation* fn = module->GetComputationWithName("Fn");
  EXPECT_THAT(fn->root_instruction(),
              GmockMatch(m::Add(m::Constant(), m::Parameter(1))));
}

TEST_F(HloConstantFoldingTest, InterproceduralMultipleCallsites) {
  const char* const kModuleStr = R"(
    HloModule test

    Fn {
      param0 = f32[8] parameter(0)
      param1 = f32[8] parameter(1)
      iota = f32[8] iota(), iota_dimension=0
      add.0 = f32[8] add(param0, iota)
      ROOT add.1 = f32[8] add(add.0, param1)
    }

    ENTRY entry {
      entry.param0 = f32[8] parameter(0)
      entry.param1 = f32[8] parameter(1)
      constant.0 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -1})
      call.0 = f32[8] call(constant.0, entry.param0), to_apply=Fn
      call.1 = f32[8] call(constant.0, entry.param1), to_apply=Fn
      ROOT add = f32[8] add(call.0, call.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  HloComputation* fn = module->GetComputationWithName("Fn");
  EXPECT_THAT(fn->root_instruction(),
              GmockMatch(m::Add(m::Constant(), m::Parameter(1))));
}

TEST_F(HloConstantFoldingTest,
       InterproceduralMultipleCallsitesDifferentConstants) {
  const char* const kModuleStr = R"(
    HloModule test

    Fn {
      param0 = f32[8] parameter(0)
      param1 = f32[8] parameter(1)
      iota = f32[8] iota(), iota_dimension=0
      add.0 = f32[8] add(param0, iota)
      ROOT add.1 = f32[8] add(add.0, param1)
    }

    ENTRY entry {
      entry.param0 = f32[8] parameter(0)
      entry.param1 = f32[8] parameter(1)
      constant.0 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -1})
      constant.1 = f32[8] constant({2, -2, 2, -2, 2, -2, 2, -2})
      call.0 = f32[8] call(constant.0, entry.param0), to_apply=Fn
      call.1 = f32[8] call(constant.1, entry.param1), to_apply=Fn
      ROOT add = f32[8] add(call.0, call.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloConstantFoldingTest, InterproceduralMultipleCallsitesSomeConstants) {
  const char* const kModuleStr = R"(
    HloModule test

    Fn {
      param0 = f32[] parameter(0)
      param1 = f32[] parameter(1)
      param2 = f32[] parameter(2)
      param3 = f32[] parameter(3)
      mul = f32[] multiply(param0, param1)
      sub = f32[] subtract(mul, param2)
      ROOT add = f32[] add(sub, param3)
    }

    ENTRY entry {
      entry.param = f32[] parameter(0)
      constant.0 = f32[] constant(1)
      constant.1 = f32[] constant(2)
      call.0 = f32[] call(constant.0, constant.1, entry.param, constant.1), to_apply=Fn
      call.1 = f32[] call(constant.0, entry.param, constant.0, constant.1), to_apply=Fn
      ROOT add = f32[] add(call.0, call.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  HloComputation* fn = module->GetComputationWithName("Fn");
  EXPECT_THAT(fn->root_instruction(),
              GmockMatch(m::Add(m::Subtract(m::Multiply(m::ConstantScalar(1),
                                                        m::Parameter(1)),
                                            m::Parameter(2)),
                                m::ConstantScalar(2))));
}

TEST_F(HloConstantFoldingTest,
       InterproceduralMultipleCallsitesSomeDifferentConstants) {
  const char* const kModuleStr = R"(
    HloModule test

    Fn {
      param0 = f32[8] parameter(0)
      param1 = f32[8] parameter(1)
      param2 = f32[8] parameter(2)
      iota = f32[8] iota(), iota_dimension=0
      add.0 = f32[8] add(param0, iota)
      add.1 = f32[8] add(add.0, param1)
      ROOT sub = f32[8] subtract(add.1, param2)
    }

    ENTRY entry {
      entry.param0 = f32[8] parameter(0)
      entry.param1 = f32[8] parameter(1)
      constant.0 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -1})
      constant.1 = f32[8] constant({2, -2, 2, -2, 2, -2, 2, -2})
      call.0 = f32[8] call(constant.0, entry.param0, constant.0), to_apply=Fn
      call.1 = f32[8] call(constant.1, entry.param1, constant.0), to_apply=Fn
      ROOT add = f32[8] add(call.0, call.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  HloComputation* fn = module->GetComputationWithName("Fn");
  EXPECT_THAT(fn->root_instruction(),
              GmockMatch(m::Subtract(m::Add(m::Add(), m::Parameter(1)),
                                     m::Constant())));
}

TEST_F(HloConstantFoldingTest, InterproceduralDeadParameter) {
  const char* const kModuleStr = R"(
    HloModule test

    Fn {
      param0 = f32[8] parameter(0)
      param1 = f32[8] parameter(1)
      iota = f32[8] iota(), iota_dimension=0
      ROOT add.1 = f32[8] add(iota, param1)
    }

    ENTRY entry {
      entry.param = f32[8] parameter(0)
      constant.0 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -1})
      ROOT call = f32[8] call(constant.0, entry.param), to_apply=Fn
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloConstantFoldingTest,
       InterproceduralMultipleCallsitesDeterministicResults) {
  const char* const kModuleStr = R"(
    HloModule test

    Fn {
      param0 = f32[8] parameter(0)
      param1 = f32[8] parameter(1)
      param2 = f32[8] parameter(2)
      ROOT add.1 = f32[8] add(param0, param1)
    }

    ENTRY entry {
      entry.param0 = f32[8] parameter(0)
      entry.param1 = f32[8] parameter(1)
      entry.param2 = f32[8] parameter(2)
      constant.0.0 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -1})
      constant.0.1 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -1})
      constant.0.2 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -1})
      constant.1 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -2})
      constant.2 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -3})
      constant.3 = f32[8] constant({1, -1, 1, -1, 1, -1, 1, -4})
      call.0 = f32[8] call(constant.0.0, constant.1, entry.param0), to_apply=Fn
      call.1 = f32[8] call(constant.0.1, constant.2, entry.param1), to_apply=Fn
      call.3 = f32[8] call(constant.0.2, constant.3, entry.param2), to_apply=Fn
      add = f32[8] add(call.0, call.1)
      ROOT add.2 = f32[8] add(add, call.3)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  HloComputation* fn = module->GetComputationWithName("Fn");
  std::string constant_name;
  for (HloInstruction* inst : fn->instructions()) {
    if (inst->opcode() == HloOpcode::kConstant) {
      constant_name = inst->name();
      break;
    }
  }
  EXPECT_GT(constant_name.size(), 0);
  // Run the pass repeatedly and check the result is deterministic.
  for (int i = 0; i < 10; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(kModuleStr));
    HloConstantFolding constant_folding;
    TF_ASSERT_OK_AND_ASSIGN(bool result,
                            RunHloPass(&constant_folding, module.get()));
    EXPECT_TRUE(result);
    HloComputation* fn = module->GetComputationWithName("Fn");
    std::string new_constant_name;
    for (HloInstruction* inst : fn->instructions()) {
      if (inst->opcode() == HloOpcode::kConstant) {
        new_constant_name = inst->name();
        break;
      }
    }
    EXPECT_EQ(constant_name, new_constant_name);
  }
}

TEST_F(HloConstantFoldingTest, DontFoldGetRngSeed) {
  const char* const kModuleStr = R"(
    HloModule test
    ENTRY entry {
      ROOT call = u64[] custom-call(), custom_call_target="GetRngSeed"
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloConstantFoldingTest, DontFoldOptimizationBarrier) {
  const char* const kModuleStr = R"(
    HloModule test
    ENTRY entry {
      c = u32[2]{0} constant({1, 2})
      ROOT ob = u32[2]{0} opt-barrier(c)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

// Folding a call whose body contains an optimization barrier would delete
// the barrier.
TEST_F(HloConstantFoldingTest, DontFoldCallContainingOptimizationBarrier) {
  const char* const kModuleStr = R"(
    HloModule test

    body {
      p = u32[2]{0} parameter(0)
      ROOT ob = u32[2]{0} opt-barrier(p)
    }

    ENTRY entry {
      c = u32[2]{0} constant({1, 2})
      ROOT call = u32[2]{0} call(c), to_apply=body
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

// Sub-byte types are stored unpacked in literals, so width-changing bitcasts
// of packed types must not fold: the raw byte copy would produce a constant
// that differs from the backend result.
TEST_F(HloConstantFoldingTest, DontFoldSubByteWidthChangingBitcasts) {
  const char* const kModuleStr = R"(
    HloModule test
    ENTRY entry {
      c32 = s32[4]{0} constant({305419896, -1, 0, 559038737})
      bc = s4[4,8]{1,0:E(4)} bitcast-convert(c32)
      c4 = s4[16]{0:E(4)} constant({0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1})
      b = s8[8]{0} bitcast(c4)
      ROOT t = (s4[4,8]{1,0:E(4)}, s8[8]{0}) tuple(bc, b)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

// The evaluator returns default layouts for reshape/concatenate results;
// folding must restore the full layout of the folded instruction, tiles
// included, so post-layout pipelines can run the pass.
TEST_F(HloConstantFoldingTest, FoldedConstantKeepsTiledLayout) {
  const char* const kModuleStr = R"(
    HloModule test
    ENTRY entry {
      c0 = u32[1]{0:T(128)} constant({1})
      c1 = u32[1]{0:T(128)} constant({2})
      cc = u32[2]{0:T(128)} concatenate(c0, c1), dimensions={0}
      p0 = u32[2]{0:T(128)} parameter(0)
      ROOT t = (u32[2]{0:T(128)}, u32[2]{0:T(128)}) tuple(cc, p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kModuleStr));
  HloConstantFolding constant_folding;
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::Constant(), op::Parameter(0)));
  EXPECT_TRUE(
      ShapeUtil::Equal(root->operand(0)->shape(), root->operand(1)->shape()));
  EXPECT_EQ(root->operand(0)->literal().Get<uint32_t>({0}), 1);
  EXPECT_EQ(root->operand(0)->literal().Get<uint32_t>({1}), 2);
}

// Options used by pipelines that fold late, after layout assignment.
HloConstantFolding::Options LateFoldingOptions() {
  HloConstantFolding::Options options;
  options.fold_float_arithmetic = false;
  options.is_layout_sensitive = true;
  return options;
}

// A per-layer PRNG key derivation (tuple shaped unstack fusion of a constant,
// xor of the halves) as exposed by post-layout passes. In layout sensitive
// mode the fusion folds through get-tuple-element rewrites, never through a
// tuple shaped constant.
TEST_F(HloConstantFoldingTest, LateOptionsFoldUnstackFusionViaGteRewrite) {
  const char* const kModuleStr = R"(
    HloModule test

    unstack_comp {
      p = u32[2]{0} parameter(0)
      sl_hi = u32[1]{0} slice(p), slice={[1:2]}
      sl_lo = u32[1]{0} slice(p), slice={[0:1]}
      ROOT t = (u32[1]{0}, u32[1]{0}) tuple(sl_hi, sl_lo)
    }

    ENTRY entry {
      key = u32[2]{0} constant({305419896, 43981})
      f = (u32[1]{0}, u32[1]{0}) fusion(key), kind=kLoop, calls=unstack_comp
      g0 = u32[1]{0} get-tuple-element(f), index=0
      g1 = u32[1]{0} get-tuple-element(f), index=1
      r0 = u32[] reshape(g0)
      r1 = u32[] reshape(g1)
      x = u32[] xor(r0, r1)
      p0 = u32[] parameter(0)
      ROOT out = u32[] add(x, p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Constant(), op::Parameter(0)));
  // 0x12345678 ^ 0x0000ABCD == 0x1234FDB5.
  EXPECT_EQ(root->operand(0)->literal().Get<uint32_t>({}), 0x1234FDB5u);
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      EXPECT_NE(instruction->opcode(), HloOpcode::kFusion);
      if (instruction->opcode() == HloOpcode::kConstant) {
        EXPECT_FALSE(instruction->shape().IsTuple());
      }
    }
  }
}

// Port of the production pattern with post-layout decorations: tiled
// layouts, pinning frontend attributes, and bitcasts instead of reshapes.
TEST_F(HloConstantFoldingTest, LateOptionsFoldProdDecoratedPattern) {
  const char* const kModuleStr = R"(
    HloModule test

    unstack_comp {
      p = u32[2]{0:T(128)} parameter(0)
      sl_hi = u32[1]{0:T(128)} slice(p), slice={[1:2]}
      sl_lo = u32[1]{0:T(128)} slice(p), slice={[0:1]}
      ROOT t = (u32[1]{0:T(128)}, u32[1]{0:T(128)}) tuple(sl_hi, sl_lo)
    }

    ENTRY entry {
      key = u32[2]{0:T(128)} constant({305419896, 43981}), frontend_attributes={xla_pinned_vmem="true"}
      f = (u32[1]{0:T(128)}, u32[1]{0:T(128)}) fusion(key), kind=kLoop, calls=unstack_comp
      g0 = u32[1]{0:T(128)} get-tuple-element(f), index=0
      g1 = u32[1]{0:T(128)} get-tuple-element(f), index=1
      b0 = u32[]{:T(128)} bitcast(g0)
      b1 = u32[]{:T(128)} bitcast(g1)
      x = u32[]{:T(128)} xor(b0, b1)
      p0 = u32[]{:T(128)} parameter(0)
      ROOT out = u32[]{:T(128)} add(x, p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Constant(), op::Parameter(0)));
  EXPECT_EQ(root->operand(0)->literal().Get<uint32_t>({}), 0x1234FDB5u);
  // The folded constant keeps the tiled scalar layout.
  EXPECT_EQ(root->operand(0)->shape().layout().tiles().size(), 1);
}

TEST_F(HloConstantFoldingTest, LateOptionsDontFoldFloatArithmetic) {
  const char* const kModuleStr = R"(
    HloModule test
    ENTRY entry {
      c0 = f32[4]{0} constant({1, 2, 3, 4})
      c1 = f32[4]{0} constant({5, 6, 7, 8})
      a = f32[4]{0} add(c0, c1)
      p0 = f32[4]{0} parameter(0)
      ROOT out = f32[4]{0} add(a, p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(HloConstantFoldingTest, LateOptionsFoldFloatDataMovement) {
  const char* const kModuleStr = R"(
    HloModule test
    ENTRY entry {
      c = f32[4]{0} constant({1, 2, 3, 4})
      s = f32[2]{0} slice(c), slice={[2:4]}
      p0 = f32[2]{0} parameter(0)
      ROOT out = f32[2]{0} add(s, p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Constant(), op::Parameter(0)));
  EXPECT_EQ(root->operand(0)->literal().Get<float>({0}), 3.0f);
  EXPECT_EQ(root->operand(0)->literal().Get<float>({1}), 4.0f);
}

// Integer to integer converts fold; converts with a float operand must not.
TEST_F(HloConstantFoldingTest, LateOptionsFoldIntButNotFloatOperandConvert) {
  const char* const kModuleStr = R"(
    HloModule test
    ENTRY entry {
      cf = f32[2]{0} constant({1.5, 2.5})
      ci = s32[2]{0} constant({7, -3})
      from_float = s32[2]{0} convert(cf)
      narrowed = s8[2]{0} convert(ci)
      widened = s32[2]{0} convert(narrowed)
      p0 = s32[2]{0} parameter(0)
      a = s32[2]{0} add(from_float, p0)
      ROOT out = s32[2]{0} add(a, widened)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Add(op::Add(op::Convert(op::Constant()), op::Parameter(0)),
                      op::Constant()));
  EXPECT_EQ(root->operand(1)->literal().Get<int32_t>({0}), 7);
  EXPECT_EQ(root->operand(1)->literal().Get<int32_t>({1}), -3);
}

TEST_F(HloConstantFoldingTest, LateOptionsDontFoldFusionWithFloatArithmetic) {
  const char* const kModuleStr = R"(
    HloModule test

    fused_comp {
      p = f32[2]{0} parameter(0)
      m = f32[2]{0} multiply(p, p)
      ROOT cv = s32[2]{0} convert(m)
    }

    ENTRY entry {
      c = f32[2]{0} constant({1.5, 2.5})
      f = s32[2]{0} fusion(c), kind=kLoop, calls=fused_comp
      p0 = s32[2]{0} parameter(0)
      ROOT out = s32[2]{0} add(f, p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

// Pad, select, and dynamic-slice move bytes without arithmetic, so they fold
// for float types too.
TEST_F(HloConstantFoldingTest, LateOptionsFoldFloatPadSelectAndDynamicSlice) {
  const char* const kModuleStr = R"(
    HloModule test
    ENTRY entry {
      c = f32[4]{0} constant({1, 2, 3, 4})
      z = f32[] constant(0)
      pd = f32[6]{0} pad(c, z), padding=1_1
      pr = pred[4]{0} constant({1, 0, 1, 0})
      c2 = f32[4]{0} constant({5, 6, 7, 8})
      sel = f32[4]{0} select(pr, c, c2)
      i = s32[] constant(1)
      ds = f32[2]{0} dynamic-slice(c2, i), dynamic_slice_sizes={2}
      p0 = f32[6]{0} parameter(0)
      p1 = f32[4]{0} parameter(1)
      p2 = f32[2]{0} parameter(2)
      a0 = f32[6]{0} add(pd, p0)
      a1 = f32[4]{0} add(sel, p1)
      a2 = f32[2]{0} add(ds, p2)
      ROOT t = (f32[6]{0}, f32[4]{0}, f32[2]{0}) tuple(a0, a1, a2)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::Add(op::Constant(), op::Parameter(0)),
                              op::Add(op::Constant(), op::Parameter(1)),
                              op::Add(op::Constant(), op::Parameter(2))));
  const Literal& pad_literal = root->operand(0)->operand(0)->literal();
  EXPECT_EQ(pad_literal.Get<float>({0}), 0.0f);
  EXPECT_EQ(pad_literal.Get<float>({1}), 1.0f);
  const Literal& select_literal = root->operand(1)->operand(0)->literal();
  EXPECT_EQ(select_literal.Get<float>({0}), 1.0f);
  EXPECT_EQ(select_literal.Get<float>({1}), 6.0f);
  const Literal& slice_literal = root->operand(2)->operand(0)->literal();
  EXPECT_EQ(slice_literal.Get<float>({0}), 6.0f);
  EXPECT_EQ(slice_literal.Get<float>({1}), 7.0f);
}

// The can_fold_shape filter skips instructions whose shapes it rejects.
TEST_F(HloConstantFoldingTest, LateOptionsRespectCanFoldShapeFilter) {
  const char* const kModuleStr = R"(
    HloModule test
    ENTRY entry {
      c0 = u32[4]{0} constant({1, 2, 3, 4})
      c1 = u32[4]{0} constant({5, 6, 7, 8})
      x = u32[4]{0} xor(c0, c1)
      ca = u32[] constant(6)
      cb = u32[] constant(3)
      xs = u32[] xor(ca, cb)
      b = u32[4]{0} broadcast(xs), dimensions={}
      ROOT out = u32[4]{0} add(x, b)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding::Options options = LateFoldingOptions();
  options.can_fold_shape = [](const Shape& shape) {
    return shape.dimensions().empty();  // Only allow scalar folds.
  };
  HloConstantFolding constant_folding(options);
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  // The vector xor is filtered out; the scalar xor folds.
  EXPECT_THAT(root, op::Add(op::Xor(op::Constant(), op::Constant()),
                            op::Broadcast(op::Constant())));
}

// A producer with a tuple shape and a non get-tuple-element user cannot be
// folded in layout sensitive mode; materializing a tuple shaped constant is
// not an option there.
TEST_F(HloConstantFoldingTest, LateOptionsDontFoldTupleWithNonGteUser) {
  const char* const kModuleStr = R"(
    HloModule test

    unstack_comp {
      p = u32[2]{0} parameter(0)
      sl_hi = u32[1]{0} slice(p), slice={[1:2]}
      sl_lo = u32[1]{0} slice(p), slice={[0:1]}
      ROOT t = (u32[1]{0}, u32[1]{0}) tuple(sl_hi, sl_lo)
    }

    ENTRY entry {
      key = u32[2]{0} constant({305419896, 43981})
      f = (u32[1]{0}, u32[1]{0}) fusion(key), kind=kLoop, calls=unstack_comp
      ROOT cc = u32[1]{0} custom-call(f), custom_call_target="Consume"
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Fusion()));
}

// A get-tuple-element of a nested tuple is itself tuple shaped and cannot be
// rewritten to a constant.
TEST_F(HloConstantFoldingTest, LateOptionsDontFoldNestedTupleGte) {
  const char* const kModuleStr = R"(
    HloModule test

    nested_comp {
      p = u32[2]{0} parameter(0)
      a = u32[2]{0} add(p, p)
      inner = (u32[2]{0}, u32[2]{0}) tuple(p, a)
      ROOT outer = ((u32[2]{0}, u32[2]{0}), u32[2]{0}) tuple(inner, a)
    }

    ENTRY entry {
      c = u32[2]{0} constant({1, 2})
      f = ((u32[2]{0}, u32[2]{0}), u32[2]{0}) fusion(c), kind=kLoop, calls=nested_comp
      g0 = (u32[2]{0}, u32[2]{0}) get-tuple-element(f), index=0
      g1 = u32[2]{0} get-tuple-element(f), index=1
      g00 = u32[2]{0} get-tuple-element(g0), index=0
      ROOT r = u32[2]{0} add(g00, g1)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

// The evaluator produces tuple leaves in default layouts; the rewritten leaf
// constants must match the replaced get-tuple-element's layout.
TEST_F(HloConstantFoldingTest, LateOptionsRelayoutTupleLeavesToGteLayout) {
  const char* const kModuleStr = R"(
    HloModule test

    add2 {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      a0 = s32[] add(p0, p2)
      a1 = s32[] add(p1, p3)
      ROOT t = (s32[], s32[]) tuple(a0, a1)
    }

    ENTRY entry {
      c0 = s32[2,2,2]{2,1,0} constant({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}})
      c1 = s32[2,2,2]{2,1,0} constant({{{10, 20}, {30, 40}}, {{50, 60}, {70, 80}}})
      z = s32[] constant(0)
      r = (s32[2,2]{0,1}, s32[2,2]{0,1}) reduce(c0, c1, z, z), dimensions={0}, to_apply=add2
      g0 = s32[2,2]{0,1} get-tuple-element(r), index=0
      g1 = s32[2,2]{0,1} get-tuple-element(r), index=1
      p0 = s32[2,2]{0,1} parameter(0)
      a = s32[2,2]{0,1} add(g0, p0)
      ROOT out = s32[2,2]{0,1} add(a, g1)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_TRUE(result);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Add(op::Add(op::Constant(), op::Parameter(0)), op::Constant()));
  const HloInstruction* leaf0 = root->operand(0)->operand(0);
  const HloInstruction* leaf1 = root->operand(1);
  for (const HloInstruction* leaf : {leaf0, leaf1}) {
    EXPECT_TRUE(LayoutUtil::Equal(leaf->shape().layout(),
                                  LayoutUtil::MakeLayout({0, 1})));
  }
  EXPECT_EQ(leaf0->literal().Get<int32_t>({0, 0}), 6);
  EXPECT_EQ(leaf0->literal().Get<int32_t>({1, 1}), 12);
  EXPECT_EQ(leaf1->literal().Get<int32_t>({0, 0}), 60);
  EXPECT_EQ(leaf1->literal().Get<int32_t>({1, 1}), 120);
}

// Rewriting a get-tuple-element that carries a control dependency to a
// constant would drop the ordering edge.
TEST_F(HloConstantFoldingTest, LateOptionsDontFoldGteWithControlDependency) {
  const char* const kModuleStr = R"(
    HloModule test

    unstack_comp {
      p = u32[2]{0} parameter(0)
      sl_hi = u32[1]{0} slice(p), slice={[1:2]}
      sl_lo = u32[1]{0} slice(p), slice={[0:1]}
      ROOT t = (u32[1]{0}, u32[1]{0}) tuple(sl_hi, sl_lo)
    }

    ENTRY entry {
      p0 = u32[1]{0} parameter(0)
      key = u32[2]{0} constant({305419896, 43981})
      gate = u32[1]{0} add(p0, p0)
      f = (u32[1]{0}, u32[1]{0}) fusion(key), kind=kLoop, calls=unstack_comp
      g0 = u32[1]{0} get-tuple-element(f), index=0
      g1 = u32[1]{0} get-tuple-element(f), index=1, control-predecessors={gate}
      x = u32[1]{0} xor(g0, g1)
      ROOT out = u32[1]{0} add(x, gate)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  HloConstantFolding constant_folding(LateFoldingOptions());
  ASSERT_OK_AND_ASSIGN(bool result,
                       RunHloPass(&constant_folding, module.get()));
  EXPECT_FALSE(result);
}

}  // namespace
}  // namespace xla
