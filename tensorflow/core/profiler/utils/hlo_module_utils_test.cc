/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/hlo_module_utils.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace profiler {
namespace {

class HloModuleUtilsTest : public xla::HloTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<xla::HloModule>> GetModuleWithStackFrames() {
    const char file_name[] = "main.py";
    const char function_name[] = "func1";
    const int line_number = 10;
    const int column_number = 5;
    const int frame_id = 1;
    const char text[] = R"(
    HloModule a_module

    ENTRY main {
      %c = s32[] constant(1)
      ROOT %result = s32[] parameter(0)
    }
    )";
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(text));

    auto module_proto = module->ToProto();
    auto index = module_proto.mutable_stack_frame_index();
    index->add_file_names(file_name);
    index->add_function_names(function_name);
    auto location = index->add_file_locations();
    location->set_file_name_id(frame_id);
    location->set_function_name_id(1);
    location->set_line(line_number);
    location->set_column(column_number);

    auto frame = index->add_stack_frames();
    frame->set_file_location_id(1);

    // Set the stack frame id of the root instruction.
    for (auto& computation : *module_proto.mutable_computations()) {
      if (computation.id() == module_proto.entry_computation_id()) {
        for (auto& instruction : *computation.mutable_instructions()) {
          if (instruction.id() == computation.root_id()) {
            instruction.mutable_metadata()->set_stack_frame_id(frame_id);
            instruction.mutable_metadata()->set_source_file(file_name);
            instruction.mutable_metadata()->set_source_line(line_number);
          }
        }
      }
    }

    return xla::HloModule::CreateFromProto(module_proto, module->config());
  }
};

TEST_F(HloModuleUtilsTest, TestGetLocationStack) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> module_with_stack_frames,
      GetModuleWithStackFrames());
  auto root_instruction =
      module_with_stack_frames->entry_computation()->root_instruction();
  EXPECT_EQ(GetOpLocationStack(1, root_instruction), "main.py:10:5\n");
}

TEST_F(HloModuleUtilsTest, TestGetSourceInfo) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> module_with_stack_frames,
      GetModuleWithStackFrames());
  auto root_instruction =
      module_with_stack_frames->entry_computation()->root_instruction();
  auto source_info = GetSourceInfo(root_instruction);
  EXPECT_EQ(source_info.source_file, "main.py");
  EXPECT_EQ(source_info.source_line, 10);
  EXPECT_EQ(source_info.stack_frame, "main.py:10:5\n");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
