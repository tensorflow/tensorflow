// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/contrib/ffmpeg/ffmpeg_lib.h"

#include <stdlib.h>
#include <vector>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::testing::TensorFlowSrcRoot;

namespace tensorflow {
namespace ffmpeg {
namespace {

const char kTestWavFilename[] =
    "contrib/ffmpeg/testdata/mono_10khz.wav";
const char kTestMp3Filename[] =
    "contrib/ffmpeg/testdata/test_sound1.mp3";

// Set to true via a command line flag iff the test is expected to have FFmpeg
// installed.
mutex mu;
bool should_ffmpeg_be_installed GUARDED_BY(mu) = false;

string ParseTestFlags(int* argc, char** argv) {
  mutex_lock l(mu);
  std::vector<Flag> flag_list = {
      Flag("should_ffmpeg_be_installed", &should_ffmpeg_be_installed,
           "indicates that ffmpeg should be installed")};
  string usage = Flags::Usage(argv[0], flag_list);
  if (!Flags::Parse(argc, argv, flag_list)) {
    LOG(ERROR) << "\n" << usage;
    exit(2);
  }
  return usage;
}

TEST(FfmpegLibTest, TestUninstalled) {
  {
    mutex_lock l(mu);
    if (should_ffmpeg_be_installed) {
      return;
    }
    LOG(INFO) << "Assuming FFmpeg is uninstalled.";
  }

  string filename = io::JoinPath(TensorFlowSrcRoot(), kTestMp3Filename);
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

  string filename = io::JoinPath(TensorFlowSrcRoot(), kTestMp3Filename);
  std::vector<float> output_samples;
  Status status = ReadAudioFile(filename, "mp3", 5000, 1, &output_samples);
  ASSERT_TRUE(status.ok());
}

TEST(FfmpegLibTest, TestRoundTripGeneratedWav) {
  {
    mutex_lock l(mu);
    if (!should_ffmpeg_be_installed) {
      return;
    }
  }

  std::vector<float> sine_wave;
  sine_wave.reserve(20000);
  for (int i = 0; i < 20000; ++i) {
    sine_wave.push_back(std::sin(6.28 * 440.0 * i / 20000.0));
  }
  string content;
  ASSERT_TRUE(CreateAudioFile("wav", 0, 20000, 1, sine_wave, &content).ok());
  string temp_filename = GetTempFilename("wav");
  ASSERT_TRUE(WriteStringToFile(Env::Default(), temp_filename, content).ok());
  std::vector<float> roundtrip_data;
  ASSERT_TRUE(
      ReadAudioFile(temp_filename, "wav", 20000, 1, &roundtrip_data).ok());
  EXPECT_EQ(sine_wave.size(), roundtrip_data.size());
  size_t min_size = std::min(sine_wave.size(), roundtrip_data.size());
  for (size_t i = 0; i < min_size; ++i) {
    EXPECT_NEAR(sine_wave[i], roundtrip_data[i], 0.01);
    EXPECT_LE(roundtrip_data[i], 1.0);
    EXPECT_LE(-1.0, roundtrip_data[i]);
  }
}

TEST(FfmpegLibTest, TestRoundTripWav) {
  {
    mutex_lock l(mu);
    if (!should_ffmpeg_be_installed) {
      return;
    }
  }

  string filename = io::JoinPath(TensorFlowSrcRoot(), kTestWavFilename);
  std::vector<float> output_samples;
  ASSERT_TRUE(ReadAudioFile(filename, "wav", 10000, 1, &output_samples).ok());
  string original_audio;
  ASSERT_TRUE(ReadFileToString(Env::Default(), filename, &original_audio).ok());

  string written_audio;
  ASSERT_TRUE(
      CreateAudioFile("wav", 0, 10000, 1, output_samples, &written_audio).ok());

  EXPECT_EQ(original_audio, written_audio);
}

}  // namespace
}  // namespace ffmpeg
}  // namespace tensorflow

int main(int argc, char **argv) {
  tensorflow::string usage = tensorflow::ffmpeg::ParseTestFlags(&argc, argv);
  testing::InitGoogleTest(&argc, argv);
  if (argc != 1) {
    LOG(ERROR) << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
