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
#include <vector>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::testing::TensorFlowSrcRoot;

namespace tensorflow {
namespace ffmpeg {
namespace {

const char kTestSoundFilename[] =
    "contrib/ffmpeg/kernels/testdata/test_sound1.mp3";

// Set to true via a command line flag iff the test is expected to have FFmpeg
// installed.
mutex mu;
bool should_ffmpeg_be_installed GUARDED_BY(mu) = false;

void ParseTestFlags(int* argc, char** argv) {
  mutex_lock l(mu);
  CHECK(ParseFlags(argc, argv, {Flag("should_ffmpeg_be_installed",
                                     &should_ffmpeg_be_installed)}));
}

TEST(FfmpegLibTest, TestUninstalled) {
  {
    mutex_lock l(mu);
    if (should_ffmpeg_be_installed) {
      return;
    }
    LOG(INFO) << "Assuming FFmpeg is uninstalled.";
  }

  string filename = io::JoinPath(TensorFlowSrcRoot(), kTestSoundFilename);
  std::vector<float> output_samples;
  Status status = ReadAudioFile(filename, "mp3", 5000, 1, &output_samples);
  ASSERT_EQ(status.code(), error::Code::NOT_FOUND);
}

TEST(FfmpegLibTest, TestInstalled) {
  {
    mutex_lock l(mu);
    if (!should_ffmpeg_be_installed) {
      return;
    }
    LOG(INFO) << "Assuming FFmpeg is installed.";
  }

  string filename = io::JoinPath(TensorFlowSrcRoot(), kTestSoundFilename);
  std::vector<float> output_samples;
  Status status = ReadAudioFile(filename, "mp3", 5000, 1, &output_samples);
  ASSERT_TRUE(status.ok());
}

}  // namespace
}  // ffmpeg
}  // tensorflow

int main(int argc, char **argv) {
  tensorflow::ffmpeg::ParseTestFlags(&argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
