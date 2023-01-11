/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/file_system.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {

class DumpMlirPassesTest : public HloTestBase {};

// Ensures that specifying the option --xla_gpu_dump_llvmir also dumps the
// intermediate IR of the run mlir passes. This is tested by compiling an
// HLO snippet that should trigger the mlir pipeline and then checking for the
// existance of a dump file <modulename>.mlir-passes.log.
XLA_TEST_F(DumpMlirPassesTest, CompileSimpleCWiseTest) {
  std::string hlo_text = R"(
  HloModule m1, entry_computation_layout={(f32[3,3]{1,0})->f32[3,3]{1,0}}
  ENTRY m1 {
    arg0.1 = f32[3,3]{1,0} parameter(0), parameter_replication={false}
    constant.4 = f32[] constant(42.0)
    broadcast.5 = f32[3,3]{1,0} broadcast(constant.4), dimensions={}
    ROOT multiply.6 = f32[3,3]{1,0} multiply(arg0.1, broadcast.5)
})";
  auto config = GetModuleConfigForTest(/*replica_count=*/2,
                                       /*num_partitions=*/1);
  auto debug_options = config.debug_options();
  debug_options.set_xla_gpu_dump_llvmir(true);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      test_runner_.CreateExecutable(std::move(module), true));

  const std::string basename = absl::StrCat(
      absl::string_view(tsl::io::Basename(executable->module().name())),
      ".mlir-passes.log");
  std::string outputs_dir;
  tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir);
  std::string path = tsl::io::JoinPath(outputs_dir, basename);

  tsl::FileSystem *fs;
  TF_ASSERT_OK(tsl::Env::Default()->GetFileSystemForFile(path, &fs));
  TF_ASSERT_OK(fs->FileExists(path));
}

}  // namespace xla
