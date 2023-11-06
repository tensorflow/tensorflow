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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "tsl/platform/path.h"
#include "tsl/platform/subprocess.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class HloExpandTest : public ::testing::Test {
 protected:
  void HloOpt(std::vector<std::string>& additional_flags) {
    std::string hlo_opt_bin =
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools", "hlo-expand");

    tsl::SubProcess proc;
    std::vector<std::string> argv = {hlo_opt_bin};
    argv.insert(argv.end(), additional_flags.begin(), additional_flags.end());
    proc.SetProgram(hlo_opt_bin, argv);
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

TEST_F(HloExpandTest, CholeskyHlo) {
  std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                                           "tests", "cholesky.hlo");
  std::vector<std::string> additional_flags = {"--input_format=hlo", hlo_path};
  HloOpt(additional_flags);

  const std::string& expected_hlo_string =
      R"(HloModule main, entry_computation_layout={()->f64[3,3]{1,0}}

ENTRY %main.3 () -> f64[3,3] {
  %constant.1 = f64[3,3]{1,0} constant({ { 1, 2, 3 }, { 2, 20, 26 }, { 3, 26, 70 } })
  ROOT %cholesky.2 = f64[3,3]{1,0} cholesky(f64[3,3]{1,0} %constant.1), lower=true
})";

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 0);
  EXPECT_THAT(stdout_output_, testing::HasSubstr(expected_hlo_string));
}

TEST_F(HloExpandTest, CholeskyExpanderHlo) {
  std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                                           "tests", "cholesky.hlo");
  std::vector<std::string> additional_flags = {"--input_format=hlo", hlo_path,
                                               "--expand_all"};
  HloOpt(additional_flags);

  const std::string& expected_hlo_string = "%xla.cholesky_f64";

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 0);
  EXPECT_THAT(stdout_output_, testing::HasSubstr(expected_hlo_string));
}

TEST_F(HloExpandTest, InvalidArgc) {
  std::vector<std::string> additional_flags = {"--input_format=hlo", "foo",
                                               "bar", "baz"};
  HloOpt(additional_flags);

  const std::string& expected_string =
      "Cannot parse more than one argument. See usage below:";

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 1);
  EXPECT_THAT(stderr_output_, testing::HasSubstr(expected_string));
}

TEST_F(HloExpandTest, InvalidInputFileExtension) {
  std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                                           "tests", "foo.bar");
  std::vector<std::string> additional_flags = {hlo_path};
  HloOpt(additional_flags);

  const std::string& expected_string =
      "input_format must be specified as [hlo|pb|pbtxt].";

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 1);
  EXPECT_THAT(stderr_output_, testing::HasSubstr(expected_string));
}

TEST_F(HloExpandTest, InvalidInputFormat) {
  std::vector<std::string> additional_flags = {"--input_format=foo"};
  HloOpt(additional_flags);

  const std::string& expected_string =
      "input_format must be specified as [hlo|pb|pbtxt].";

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 1);
  EXPECT_THAT(stderr_output_, testing::HasSubstr(expected_string));
}

TEST_F(HloExpandTest, InvalidOutputFileExtension) {
  std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                                           "tests", "cholesky.hlo");
  std::string output_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                                              "tools", "tests", "foo.bar");
  std::vector<std::string> additional_flags = {"--input_format=", hlo_path,
                                               "--output_file=" + output_path};
  HloOpt(additional_flags);

  const std::string& expected_string =
      "output_format must be specified as [hlo|pb|pbtxt].";

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 1);
  EXPECT_THAT(stderr_output_, testing::HasSubstr(expected_string));
}

TEST_F(HloExpandTest, InvalidOutputFormat) {
  std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                                           "tests", "cholesky.hlo");
  std::vector<std::string> additional_flags = {"--input_format=", hlo_path,
                                               "--output_format=foo"};
  HloOpt(additional_flags);

  const std::string& expected_string =
      "output_format must be specified as [hlo|pb|pbtxt].";

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 1);
  EXPECT_THAT(stderr_output_, testing::HasSubstr(expected_string));
}

TEST_F(HloExpandTest, InvalidFile) {
  std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                                           "tests", "foo.bar");
  std::vector<std::string> additional_flags = {"--input_format=hlo", hlo_path};
  HloOpt(additional_flags);

  const std::string& expected_string = "Try: hlo-expand --help";

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 1);
  EXPECT_THAT(stderr_output_, testing::HasSubstr(expected_string));
}

TEST_F(HloExpandTest, UnsupportedOutputFormat) {
  std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                                           "tests", "cholesky.hlo");
  std::vector<std::string> additional_flags = {"--input_format=hlo",
                                               "--output_format=pb", hlo_path};
  HloOpt(additional_flags);

  const std::string& expected_string =
      "Printing to stdout must specify supported output_format=[hlo|pbtxt].";

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 1);
  EXPECT_THAT(stderr_output_, testing::HasSubstr(expected_string));
}

TEST_F(HloExpandTest, VerificationFailure) {
  std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                                           "tests", "invalid_concat.hlo");
  std::vector<std::string> additional_flags = {"--verify_hlo", hlo_path};
  HloOpt(additional_flags);

  const std::string& expected_string =
      "Cannot concatenate arrays that differ in dimensions";

  EXPECT_TRUE(exited_normally_);
  EXPECT_EQ(exit_status_, 1);
  EXPECT_THAT(stderr_output_, testing::HasSubstr(expected_string));
}

}  // namespace
}  // namespace xla
