// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/contrib/ffmpeg/kernels/ffmpeg_lib.h"

#include <stdlib.h>

#include "tensorflow/core/lib/core/command_line_flags.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

TF_DEFINE_bool(
    should_ffmpeg_be_installed, false,
    "Is it expected that FFmpeg is installed on the machine running the test?");

using tensorflow::testing::TensorFlowSrcRoot;

namespace tensorflow {
namespace ffmpeg {
namespace {

const char kTestSoundFilename[] =
    "contrib/ffmpeg/kernels/testdata/test_sound.mp3";

TEST(FfmpegLibTest, TestUninstalled) {
  if (FLAGS_should_ffmpeg_be_installed) {
    return;
  }
  LOG(INFO) << "Assuming FFmpeg is uninstalled.";

  string filename = io::JoinPath(TensorFlowSrcRoot(), kTestSoundFilename);
  std::vector<float> output_samples;
  Status status = ReadAudioFile(filename, "mp3", 5000, 1, &output_samples);
  ASSERT_EQ(status.code(), error::Code::NOT_FOUND);
}

TEST(FfmpegLibTest, TestInstalled) {
  if (!FLAGS_should_ffmpeg_be_installed) {
    return;
  }
  LOG(INFO) << "Assuming FFmpeg is installed.";

  string filename = io::JoinPath(TensorFlowSrcRoot(), kTestSoundFilename);
  std::vector<float> output_samples;
  Status status = ReadAudioFile(filename, "mp3", 5000, 1, &output_samples);
  ASSERT_TRUE(status.ok());
}

}  // namespace
}  // ffmpeg
}  // tensorflow
