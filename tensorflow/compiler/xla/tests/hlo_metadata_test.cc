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

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/local_client_test_base.h"

namespace xla {
namespace {

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
      local_client_->Compile(builder.Build().ValueOrDie(),
                             {&argument_layout, &argument_layout},
                             ExecutableBuildOptions()));

  auto instruction = executables[0]
                         ->executable()
                         ->module()
                         .entry_computation()
                         ->root_instruction();
  EXPECT_EQ("add", instruction->metadata().op_type());
  EXPECT_EQ("my_sum_op", instruction->metadata().op_name());
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
      local_client_->Compile(builder.Build().ValueOrDie(),
                             {&argument_layout, &argument_layout},
                             ExecutableBuildOptions()));

  auto instruction = executables[0]
                         ->executable()
                         ->module()
                         .entry_computation()
                         ->root_instruction();
  // We expect these to be empty (no metadata set).
  EXPECT_EQ("", instruction->metadata().op_type());
  EXPECT_EQ("", instruction->metadata().op_name());
}

}  // namespace
}  // namespace xla
