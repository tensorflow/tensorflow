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

#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/service/local_service.h"
#include "xla/test_helpers.h"
#include "xla/tests/local_client_test_base.h"

namespace xla {
namespace {

using ::testing::StrEq;

class HloMetadataTest : public LocalClientTestBase {
 protected:
  HloMetadataTest() {
    metadata_.set_op_type("add");
    metadata_.set_op_name("my_sum_op");
  }

  void BuildAddComputation(XlaBuilder* builder) {
    auto x = Parameter(builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
    auto y = Parameter(builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
    Add(x, y);
  }

  OpMetadata metadata_;
};

TEST_F(HloMetadataTest, MetadataPropagation) {
  XlaBuilder builder("add");
  builder.SetOpMetadata(metadata_);
  BuildAddComputation(&builder);
  builder.ClearOpMetadata();

  Shape argument_layout = ShapeUtil::MakeShape(F32, {});
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(),
                             {&argument_layout, &argument_layout},
                             ExecutableBuildOptions()));

  auto instruction = executables[0]
                         ->executable()
                         ->module()
                         .entry_computation()
                         ->root_instruction();
  EXPECT_THAT(instruction->metadata().op_type(), StrEq("add"));
  EXPECT_THAT(instruction->metadata().op_name(), StrEq("my_sum_op"));
}

TEST_F(HloMetadataTest, MetadataClearing) {
  XlaBuilder builder("add");
  builder.SetOpMetadata(metadata_);
  // Some other pretend computation here.
  builder.ClearOpMetadata();
  BuildAddComputation(&builder);

  Shape argument_layout = ShapeUtil::MakeShape(F32, {});
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(builder.Build().value(),
                             {&argument_layout, &argument_layout},
                             ExecutableBuildOptions()));

  auto instruction = executables[0]
                         ->executable()
                         ->module()
                         .entry_computation()
                         ->root_instruction();
  EXPECT_THAT(instruction->metadata().op_type(), StrEq(""));
  EXPECT_THAT(instruction->metadata().op_name(), StrEq(""));
}

}  // namespace
}  // namespace xla
