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
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
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

}  // namespace
}  // namespace xla
