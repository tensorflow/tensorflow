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

#include "xla/hlo/builder/lib/tuple.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using TupleTest = HloHardwareIndependentTestBase;

TEST_F(TupleTest, DisassembleAssemble) {
  XlaBuilder builder(TestName());

  Shape shape = ShapeUtil::MakeTupleShape({
      ShapeUtil::MakeShape(S32, {3}),
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(S32, {4}), ShapeUtil::MakeShape(S32, {5})}),
      ShapeUtil::MakeShape(S32, {6}),
  });
  Literal input = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateFullWithDescendingLayout({3}, int32_t{42}),
      LiteralUtil::MakeTupleOwned(
          LiteralUtil::CreateFullWithDescendingLayout({4}, int32_t{43}),
          LiteralUtil::CreateFullWithDescendingLayout({5}, int32_t{44})),
      LiteralUtil::CreateFullWithDescendingLayout({6}, int32_t{45}));

  XlaOp param = Parameter(&builder, 0, shape, "param");
  TF_ASSERT_OK_AND_ASSIGN(ShapeTree<XlaOp> disassembled_tuple,
                          DisassembleTuple(param));
  int32_t addend = 1;
  disassembled_tuple.ForEachMutableElement([&](const ShapeIndex& index,
                                               XlaOp* element) {
    const Shape& subshape = ShapeUtil::GetSubshape(shape, index);
    if (subshape.IsArray()) {
      *element = Add(
          *element,
          ConstantLiteral(&builder, LiteralUtil::CreateFullWithDescendingLayout(
                                        subshape.dimensions(), addend)));
      ++addend;
    }
  });
  AssembleTuple(&builder, std::move(disassembled_tuple));

  static constexpr absl::string_view kExpectedHlo = R"hlo(
HloModule DisassembleAssemble.1, entry_computation_layout={((s32[3]{0}, (s32[4]{0}, s32[5]{0}), s32[6]{0}))->(s32[3]{0}, (s32[4]{0}, s32[5]{0}), s32[6]{0})}

ENTRY %DisassembleAssemble.1 (param.1: (s32[3], (s32[4], s32[5]), s32[6])) -> (s32[3], (s32[4], s32[5]), s32[6]) {
  %param.1 = (s32[3]{0}, (s32[4]{0}, s32[5]{0}), s32[6]{0}) parameter(0)
  %get-tuple-element.5 = s32[3]{0} get-tuple-element(%param.1), index=0
  %constant.4 = s32[] constant(1)
  %broadcast.4 = s32[3]{0} broadcast(%constant.4), dimensions={}
  %add.4 = s32[3]{0} add(%get-tuple-element.5, %broadcast.4)
  %get-tuple-element.6 = (s32[4]{0}, s32[5]{0}) get-tuple-element(%param.1), index=1
  %get-tuple-element.7 = s32[4]{0} get-tuple-element(%get-tuple-element.6), index=0
  %constant.5 = s32[] constant(2)
  %broadcast.5 = s32[4]{0} broadcast(%constant.5), dimensions={}
  %add.5 = s32[4]{0} add(%get-tuple-element.7, %broadcast.5)
  %get-tuple-element.8 = s32[5]{0} get-tuple-element(%get-tuple-element.6), index=1
  %constant.6 = s32[] constant(3)
  %broadcast.6 = s32[5]{0} broadcast(%constant.6), dimensions={}
  %add.6 = s32[5]{0} add(%get-tuple-element.8, %broadcast.6)
  %tuple.2 = (s32[4]{0}, s32[5]{0}) tuple(%add.5, %add.6)
  %get-tuple-element.9 = s32[6]{0} get-tuple-element(%param.1), index=2
  %constant.7 = s32[] constant(4)
  %broadcast.7 = s32[6]{0} broadcast(%constant.7), dimensions={}
  %add.7 = s32[6]{0} add(%get-tuple-element.9, %broadcast.7)
  ROOT %tuple.3 = (s32[3]{0}, (s32[4]{0}, s32[5]{0}), s32[6]{0}) tuple(%add.4, %tuple.2, %add.7)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> expected_hlo_module,
                       ParseAndReturnVerifiedModule(kExpectedHlo));
  ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  DebugOptions debug_options;
  ASSERT_OK_AND_ASSIGN(HloModuleConfig config,
                       HloModule::CreateModuleConfigFromProto(
                           computation.proto(), debug_options));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> actual_hlo_module,
                       HloModule::CreateFromProto(computation.proto(), config));
  EXPECT_EQ(actual_hlo_module->GetFingerprint128(),
            expected_hlo_module->GetFingerprint128());
}

}  // namespace
}  // namespace xla
