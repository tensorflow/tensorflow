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

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::HasSubstr;

class FloatCheckDeviceE2eTest : public ::testing::Test {
 protected:
  void SetUp() override {
    env_ = tsl::Env::Default();
    run_hlo_module_bin_ = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                                            "run_hlo_module");
    ASSERT_TRUE(env_->FileExists(run_hlo_module_bin_).ok())
        << "run_hlo_module binary not found at " << run_hlo_module_bin_;
  }

  tsl::Env* env_;
  std::string run_hlo_module_bin_;
};

TEST_F(FloatCheckDeviceE2eTest, NanInDumpModeShouldDumpAndReproduce) {
  std::string tmp_dir;
  ASSERT_TRUE(env_->LocalTempFilename(&tmp_dir));
  TF_ASSERT_OK(env_->RecursivelyCreateDir(tmp_dir));

  // Set up a test HLO module that produces NaNs depending on a particular set
  // of inputs, to prevent the compiler from optimizing it out to a constant.
  std::string hlo_string = R"(
HloModule test_module

ENTRY main {
  p0 = f32[] parameter(0)
  p0_broadcast = f32[1024] broadcast(p0), dimensions={}
  zero = f32[] constant(0)
  zero_init = f32[1024] broadcast(zero), dimensions={}
  ROOT div = f32[1024] divide(zero_init, p0_broadcast)
}
)";
  std::string hlo_path = tsl::io::JoinPath(tmp_dir, "crashing_module.hlo");
  TF_ASSERT_OK(tsl::WriteStringToFile(env_, hlo_path, hlo_string));

  std::string inputs_pbtxt = R"(iterations {
  arguments {
    shape {
      element_type: F32
      layout {}
    }
    f32s: 0.0
  }
}
)";
  std::string inputs_path = tsl::io::JoinPath(tmp_dir, "inputs.pbtxt");
  TF_ASSERT_OK(tsl::WriteStringToFile(env_, inputs_path, inputs_pbtxt));

  // Run run_hlo_module under DETECTION_MODE_DUMP (dump) on GPU (CUDA).
  // We expect it to terminate with a non-zero exit code due to LOG(FATAL).
  std::vector<std::string> run_args = {
      run_hlo_module_bin_,
      "--platform=cuda",
      "--reference_platform=",
      "--input_format=hlo",
      hlo_path,
      absl::StrCat("--input_literals_file=", inputs_path),
      absl::StrCat("--xla_dump_to=", tmp_dir),
      "--xla_gpu_detect_nan=dump",
  };

  tsl::SubProcess proc;
  proc.SetProgram(run_hlo_module_bin_, run_args);
  proc.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  proc.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  ASSERT_TRUE(proc.Start());

  std::string stdout_output;
  std::string stderr_output;
  proc.Communicate(nullptr, &stdout_output, &stderr_output);

  LOG(INFO) << "run_hlo_module stdout:\n" << stdout_output;
  LOG(INFO) << "run_hlo_module stderr:\n" << stderr_output;

  // We expect it to crash (LOG(FATAL) exits with non-zero).
  EXPECT_FALSE(proc.exit_normal());
  EXPECT_THAT(stderr_output, HasSubstr("Float check crash dump generated"));
  EXPECT_THAT(stderr_output,
              HasSubstr(tsl::io::JoinPath(tmp_dir, "crash_dump")));

  // Verify that crash dump snapshot artifact was created.
  std::string crash_dump_dir = tsl::io::JoinPath(tmp_dir, "crash_dump");
  std::vector<std::string> files;
  TF_ASSERT_OK(env_->GetChildren(crash_dump_dir, &files));

  std::string snapshot_file;
  for (const auto& file : files) {
    if (absl::EndsWith(file, ".snapshot.pb")) {
      snapshot_file = tsl::io::JoinPath(crash_dump_dir, file);
      break;
    }
  }
  ASSERT_FALSE(snapshot_file.empty())
      << "No snapshot file found in " << crash_dump_dir;

  uint64_t file_size = 0;
  TF_ASSERT_OK(env_->GetFileSize(snapshot_file, &file_size));
  EXPECT_GT(file_size, 0);

  // Verify the crash dump artifacts can be used to reproduce the crash.
  std::vector<std::string> reproduce_args = {
      run_hlo_module_bin_,
      "--platform=cuda",
      "--reference_platform=",
      "--input_format=pb",
      snapshot_file,
      absl::StrCat("--xla_dump_to=", tmp_dir),
      "--xla_gpu_detect_nan=fail",
  };

  tsl::SubProcess proc_repr;
  proc_repr.SetProgram(run_hlo_module_bin_, reproduce_args);
  proc_repr.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  proc_repr.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  ASSERT_TRUE(proc_repr.Start());

  std::string stdout_output_repr;
  std::string stderr_output_repr;
  proc_repr.Communicate(nullptr, &stdout_output_repr, &stderr_output_repr);

  LOG(INFO) << "reproduce stdout:\n" << stdout_output_repr;
  LOG(INFO) << "reproduce stderr:\n" << stderr_output_repr;

  EXPECT_FALSE(proc_repr.exit_normal());
  EXPECT_THAT(stderr_output_repr, HasSubstr("Float check failed, aborting."));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
