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

// TODO(ddunleavy): test something more specific.

std::string PathToAddHlo() {
  return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools", "data",
                           "add.hlo");
}

TEST(ReplayComputation, AddHloHost) {
  // Get relevant paths to run_hlo_module and add.hlo
  std::string replay_computation_bin = tsl::io::JoinPath(
      tsl::testing::XlaSrcRoot(), "tools", "replay_computation_cpu");

  tsl::SubProcess proc;
  proc.SetProgram(replay_computation_bin,
                  {replay_computation_bin, PathToAddHlo(), "--use_fake_data"});
  proc.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  proc.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  std::string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  ASSERT_THAT(err, testing::HasSubstr("iteration complete"));
}

TEST(ReplayComputation, AddHloInterpreter) {
  // Get relevant paths to run_hlo_module and add.hlo
  std::string replay_computation_bin = tsl::io::JoinPath(
      tsl::testing::XlaSrcRoot(), "tools", "replay_computation_interpreter");

  tsl::SubProcess proc;
  proc.SetProgram(replay_computation_bin,
                  {replay_computation_bin, PathToAddHlo(), "--use_fake_data"});
  proc.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  proc.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  std::string out, err;
  int status = proc.Communicate(nullptr, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  ASSERT_THAT(err, testing::HasSubstr("iteration complete"));
}

}  // namespace
}  // namespace xla
