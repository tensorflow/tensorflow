/* Copyright 2026 The OpenXLA Authors.

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
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"

namespace xla::gpu {

namespace {

const char* input_file = "";
float abs_error_bound = 0.0;
float rel_error_bound = 0.0;

using CorrectnessTest = HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(CorrectnessTest, RunAndCompare) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          LoadModuleFromFile(input_file));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{abs_error_bound, rel_error_bound}));
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char* argv[]) {
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("abs_error_bound", &xla::gpu::abs_error_bound,
                "Absolute error bound."),
      tsl::Flag("rel_error_bound", &xla::gpu::rel_error_bound,
                "Relative error bound.")};

  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parseResult = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parseResult || argc != 2) {
    LOG(ERROR) << "\n" << usage;
    return 1;
  }

  xla::gpu::input_file = argv[1];
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
