/* Copyright 2021 The OpenXLA Authors.

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
#include <vector>

#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_parser.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/subprocess.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class RunHloModuleTest : public ::testing::Test {
 protected:
  void RunHlo(const std::string& file_name,
              std::vector<std::string> extra_args = {}) {
    std::string run_hlo_module_bin = tsl::io::JoinPath(
        tsl::testing::XlaSrcRoot(), "tools", "run_hlo_module");

    std::string hlo_path = file_name[0] == '/'
                               ? file_name
                               : tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                                                   "tools", "data", file_name);

    tsl::SubProcess proc;
    std::vector<std::string> args = {run_hlo_module_bin, hlo_path,
                                     "--platform=Host"};
    args.insert(args.end(), extra_args.begin(), extra_args.end());
    proc.SetProgram(run_hlo_module_bin, args);
    proc.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    proc.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    EXPECT_TRUE(proc.Start());

    stdout_output_ = stderr_output_ = "";
    int status = proc.Communicate(nullptr, &stdout_output_, &stderr_output_);
#if defined(_WIN32) || defined(_WIN64)
    exited_normally_ = (status == 0);
    exit_status_ = status;
#else
    exited_normally_ = WIFEXITED(status);
    exit_status_ = exited_normally_ ? WEXITSTATUS(status) : -1;
#endif  // (_WIN32) || defined (_WIN64)
    VLOG(2) << "stdout:\n" << stdout_output_ << "\n";
    VLOG(2) << "stderr:\n" << stderr_output_ << "\n";
  }

  std::string stdout_output_;
  std::string stderr_output_;
  bool exited_normally_ = false;
  int exit_status_ = -1;
};

TEST_F(RunHloModuleTest, AddHlo) {
  RunHlo("add.hlo");

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 0);
  ASSERT_THAT(
      stderr_output_,
      testing::HasSubstr("Results on Host and Interpreter are close enough."));
  EXPECT_THAT(stderr_output_,
              testing::Not(testing::HasSubstr("memory allocation bug")));
}

TEST_F(RunHloModuleTest, MustAlias) {
  RunHlo("must_alias.hlo");

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 0);
  EXPECT_THAT(
      stderr_output_,
      testing::HasSubstr("Results on Host and Interpreter are close enough."));
  EXPECT_THAT(stderr_output_,
              testing::Not(testing::HasSubstr("memory allocation bug")));
}

TEST_F(RunHloModuleTest, MustAliasWithSharding) {
  RunHlo("must_alias_with_sharding.hlo");

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 255);
  EXPECT_THAT(stderr_output_,
              testing::HasSubstr("Failed to execute on Interpreter"));
  EXPECT_THAT(stderr_output_,
              testing::Not(testing::HasSubstr("memory allocation bug")));
}

TEST_F(RunHloModuleTest, AddSnapshot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(R"(
HloModule f

ENTRY f {
  arg = f32[2,2]{1,0} parameter(0)
  ROOT add_result = f32[2,2]{1,0} add(arg, arg)
})"));

  HloSnapshot snapshot;
  *snapshot.mutable_hlo()->mutable_hlo_module() = module->ToProto();
  Literal literal = LiteralUtil::CreateR2<float>({{1, 1}, {1, 1}});
  *snapshot.add_arguments() = literal.ToProto();

  tsl::Env* env = tsl::Env::Default();
  std::string snapshot_file = testing::TempDir();
  env->CreateUniqueFileName(&snapshot_file, ".pb");
  TF_ASSERT_OK(tsl::WriteBinaryProto(env, snapshot_file, snapshot));

  RunHlo(snapshot_file, {"--print_literals"});

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 0);
  ASSERT_THAT(
      stderr_output_,
      testing::HasSubstr("Results on Host and Interpreter are close enough."));
  // Test that the arguments in the HloSnapshot are used by checking output.
  ASSERT_THAT(stdout_output_, testing::HasSubstr(R"(
** Result with test runner Host **
f32[2,2] {
  { 2, 2 },
  { 2, 2 }
})"));
}

}  // namespace
}  // namespace xla
