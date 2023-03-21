/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/subprocess.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

class RunHloModuleTest : public ::testing::Test {
 protected:
  void RunHlo(const std::string& file_name) {
    std::string run_hlo_module_bin = tsl::io::JoinPath(
        tsl::testing::XlaSrcRoot(), "tools", "run_hlo_module");

    std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                                             "tools", "data", file_name);

    tsl::SubProcess proc;
    proc.SetProgram(run_hlo_module_bin,
                    {run_hlo_module_bin, hlo_path, "--platform=Host"});
    proc.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    proc.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    EXPECT_TRUE(proc.Start());

    stdout_output_ = stderr_output_ = "";
    int status = proc.Communicate(nullptr, &stdout_output_, &stderr_output_);
    exited_normally_ = WIFEXITED(status);
    exit_status_ = exited_normally_ ? WEXITSTATUS(status) : -1;
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

}  // namespace
}  // namespace xla
