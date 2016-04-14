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

#include <errno.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace ffmpeg {
namespace {

const char kFfmpegExecutable[] = "ffmpeg";
const int32 kDefaultProbeSize = 5000000;  // 5MB

string GetTempFilename(const string& extension) {
  for (const char* dir : std::vector<const char*>(
           {getenv("TEST_TMPDIR"), getenv("TMPDIR"), getenv("TMP"), "/tmp"})) {
    if (!dir || !dir[0]) {
      continue;
    }
    struct stat statbuf;
    if (!stat(dir, &statbuf) && S_ISDIR(statbuf.st_mode)) {
      return io::JoinPath(dir, StrCat("tmp_file_", getpid(), ".", extension));
    }
  }
  LOG(FATAL) << "No temp directory found.";
}

std::vector<string> FfmpegCommandLine(const string& input_filename,
                                      const string& output_filename,
                                      const string& input_format_id,
                                      int32 samples_per_second,
                                      int32 channel_count) {
  return {
    "-nostats",  // No additional progress display.
    "-nostdin",  // No interactive commands accepted.
    "-f", input_format_id,  // eg: "mp3"
    "-probesize", StrCat(kDefaultProbeSize),
    "-i", input_filename,
    "-loglevel", "info",  // Enable verbose logging to support debugging.
    "-map_metadata", "-1",  // Copy global metadata from input to output.
    "-vn",  // No video recording.
    "-ac:a:0", StrCat(channel_count),
    "-ar:a:0", StrCat(samples_per_second),
    // Output set (in several ways) to signed 16-bit little-endian ints.
    "-codec:a:0", "pcm_s16le", "-sample_fmt", "s16", "-f", "s16le",
    "-sn",  // No subtitle recording.
    "-y",  // Overwrite output file.
    StrCat(output_filename)
  };
}

[[noreturn]] int ExecuteFfmpeg(const std::vector<string>& args) {
  std::vector<char*> args_chars;
  std::transform(args.begin(), args.end(), std::back_inserter(args_chars),
                 [](const string& s) { return const_cast<char*>(s.c_str()); });
  args_chars.push_back(nullptr);

  ::execvp(kFfmpegExecutable, args_chars.data());
  // exec only returns on error.
  const int error = errno;
  LOG(ERROR) << "FFmpeg could not be executed: " << error;
  ::_exit(error);
}

}  // namespace

Status ReadAudioFile(const string& filename,
                     const string& audio_format_id,
                     int32 samples_per_second,
                     int32 channel_count,
                     std::vector<float>* output_samples) {
  // Create an argument list.
  string output_filename = GetTempFilename(audio_format_id);
  const std::vector<string> args =
      FfmpegCommandLine(filename, output_filename, audio_format_id,
                        samples_per_second, channel_count);

  // Execute ffmpeg and report errors.
  pid_t child_pid = ::fork();
  if (child_pid < 0) {
    return Status(error::Code::UNKNOWN, StrCat("fork failed: ", errno));
  }
  if (child_pid == 0) {
    ExecuteFfmpeg(args);
  } else {
    int status_code;
    ::waitpid(child_pid, &status_code, 0);
    if (!status_code) {
      return Status::OK();
    } else {
      return Status(error::Code::NOT_FOUND,
                    StrCat("FFmpeg execution failed: ", status_code));
    }
  }
}

}  // namespace ffmpeg
}  // namespace tensorflow
