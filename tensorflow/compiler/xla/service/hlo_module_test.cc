/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {

namespace {

class HloModuleTest : public HloTestBase {
 protected:
  HloModuleTest() {}

  // Create a computation which returns a constant.
  std::unique_ptr<HloComputation> CreateConstantComputation() {
    auto builder = HloComputation::Builder("Constant");
    builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
    return builder.Build();
  }

  // Creates a computation which calls the given zero-parameter computations.
  std::unique_ptr<HloComputation> CreateCallComputation(
      tensorflow::gtl::ArraySlice<HloComputation*> computations) {
    auto builder = HloComputation::Builder("Call");
    for (auto computation : computations) {
      builder.AddInstruction(
          HloInstruction::CreateCall(r0f32_, {}, computation));
    }
    return builder.Build();
  }

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
};

TEST_F(HloModuleTest, OneComputationPostOrder) {
  // Create a module with a single computation.
  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(CreateConstantComputation());

  EXPECT_EQ(module->MakeComputationPostOrder().front(), computation);
}

TEST_F(HloModuleTest, TwoComputationsPostOrder) {
  // Create a module with two unconnected computations.
  auto module = MakeUnique<HloModule>(TestName());
  auto computation1 = module->AddEntryComputation(CreateConstantComputation());
  auto computation2 =
      module->AddEmbeddedComputation(CreateConstantComputation());

  EXPECT_MATCH(
      testing::ListToVec<HloComputation*>(module->MakeComputationPostOrder()),
      testing::UnorderedMatcher<HloComputation*>(computation1, computation2));
}

TEST_F(HloModuleTest, DiamondComputationsPostOrder) {
  // Create a module with a diamond call graph of computations.
  auto module = MakeUnique<HloModule>(TestName());
  auto computation1 =
      module->AddEmbeddedComputation(CreateConstantComputation());
  auto computation2 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation3 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation4 = module->AddEntryComputation(
      CreateCallComputation({computation2, computation3}));

  auto post_order = module->MakeComputationPostOrder();
  EXPECT_MATCH(testing::ListToVec<HloComputation*>(post_order),
               testing::UnorderedMatcher<HloComputation*>(
                   computation1, computation2, computation3, computation4));
  EXPECT_EQ(post_order.back(), computation4);
  EXPECT_EQ(post_order.front(), computation1);
}

}  // namespace

}  // namespace xla
