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

#include <memory>
#include "tensorflow/compiler/xla/service/slice_delaying.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class SliceDelayingTest : public HloTestBase {
 protected:
  SliceDelayingTest() {}
};

TEST_F(SliceDelayingTest, Basic) {
  // Verify that no dead code is removed from a computation with no dead code.
  auto builder = HloComputation::Builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {20, 10});
  auto param_0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, param_shape, "param_0"));
  auto param_1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, param_shape, "param_1"));
  Shape slice_shape_0 = ShapeUtil::MakeShape(F32, {12, 10});
  Shape slice_shape_1 = ShapeUtil::MakeShape(F32, {8, 10});
  auto slice_00 = builder.AddInstruction(
      HloInstruction::CreateSlice(slice_shape_0, param_0,
      /*start_indices=*/{0, 0}, /*limit_indices=*/{12, 10},
      /*strides=*/{1, 1}));
  auto slice_01 = builder.AddInstruction(
      HloInstruction::CreateSlice(slice_shape_1, param_0,
      /*start_indices=*/{12, 0}, /*limit_indices=*/{20, 10},
      /*strides=*/{1, 1}));
  auto slice_10 = builder.AddInstruction(
      HloInstruction::CreateSlice(slice_shape_0, param_1,
      /*start_indices=*/{0, 0}, /*limit_indices=*/{12, 10},
      /*strides=*/{1, 1}));
  auto slice_11 = builder.AddInstruction(
      HloInstruction::CreateSlice(slice_shape_1, param_1,
      /*start_indices=*/{12, 0}, /*limit_indices=*/{20, 10},
      /*strides=*/{1, 1}));
  auto add_0 = builder.AddInstruction(HloInstruction::CreateBinary(
      slice_shape_0, HloOpcode::kAdd, slice_00, slice_10));
  auto add_1 = builder.AddInstruction(HloInstruction::CreateBinary(
      slice_shape_1, HloOpcode::kAdd, slice_01, slice_11));
  builder.AddInstruction(HloInstruction::CreateTuple({add_0, add_1}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(9, computation->instruction_count());

  SliceDelaying slice_delaying;
  EXPECT_TRUE(slice_delaying.Run(module.get()).ValueOrDie());
  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());

  EXPECT_EQ(6, computation->instruction_count());
}

}  // namespace xla
