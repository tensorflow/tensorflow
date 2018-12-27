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

#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_support_checker.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

class GpuHloSupportCheckerTest : public HloTestBase {
 protected:
  GpuHloSupportChecker& checker() { return checker_; }

 private:
  GpuHloSupportChecker checker_;
};

TEST_F(GpuHloSupportCheckerTest, Add) {
  HloComputation::Builder builder(TestName());
  const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param1"));
  builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param0, param1));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(checker().Run(module.get()).status());
}

TEST_F(GpuHloSupportCheckerTest, SparseUnimplemented) {
  HloComputation::Builder builder(TestName());
  const Shape sparse_shape = ShapeUtil::MakeShapeWithSparseLayout(F32, {10}, 2);
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, sparse_shape, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, sparse_shape, "param1"));
  builder.AddInstruction(HloInstruction::CreateBinary(
      sparse_shape, HloOpcode::kAdd, param0, param1));
  // Since verifier is reporting sparse layouts as errors, we should
  // use a regular HloModule instead of VerifiedHloModule to avoid
  // verifier errors being triggered in the destructor.
  auto module = CreateNewUnverifiedModule();
  module->AddEntryComputation(builder.Build());

  Status status = checker().Run(module.get()).status();
  ASSERT_EQ(status.code(), tensorflow::error::UNIMPLEMENTED);
  EXPECT_THAT(status.error_message(),
              HasSubstr("GPU backend does not support"));
  EXPECT_THAT(status.error_message(),
              HasSubstr(ShapeUtil::HumanStringWithLayout(sparse_shape)));
}

}  // namespace
}  // namespace xla
