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

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tsl/platform/path.h"
#include "tsl/platform/subprocess.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(InteractiveGraphviz, CPU) {
  // Get path to executable
  std::string interactive_graphviz_bin = tsl::io::JoinPath(
      tsl::testing::XlaSrcRoot(), "tools", "interactive_graphviz");

  // Make string containing "--hlo_text=path/to/tools/add.hlo"
  std::string hlo_text_flag = "--hlo_text=";
  absl::StrAppend(&hlo_text_flag,
                  tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools", "data",
                                    "add.hlo"));

  // We need to specify the platform here to make sure that the binary can
  // compile the HLO. This makes sure everything necessary for compiling HLOs
  // was actually linked in. Just compiling the interactive_graphviz binary on
  // its own doesn't prove this.
  std::vector<std::string> args = {interactive_graphviz_bin, hlo_text_flag,
                                   "--platform=Host"};

  // Logging to stderr is the default externally.
  if (!tsl::testing::kIsOpenSource) args.push_back("--logtostderr");

  tsl::SubProcess proc;
  proc.SetProgram(interactive_graphviz_bin, args);
  proc.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  proc.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  EXPECT_TRUE(proc.Start());

  // We just want to make sure the executable can compile the HLO we give it
  // and then exit immediately.
  std::string in = "quit\n";
  std::string out, err;

  int status = proc.Communicate(&in, &out, &err);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  ASSERT_THAT(err, testing::HasSubstr("Compiling module for Host"));
}

}  // namespace
}  // namespace xla
