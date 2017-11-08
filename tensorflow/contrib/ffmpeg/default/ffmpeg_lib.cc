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

#include <errno.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <vector>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace ffmpeg {
namespace {

const char kFfmpegExecutable[] = "ffmpeg";
const int32 kDefaultProbeSize = 5000000;  // 5MB

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

// Is a named binary installed and executable by the current process?
// Note that this is harder than it seems like it should be...
bool IsBinaryInstalled(const string& binary_name) {
  string path = ::getenv("PATH");
  for (const string& dir : str_util::Split(path, ':')) {
    const string binary_path = io::JoinPath(dir, binary_name);
    char absolute_path[PATH_MAX + 1];
    if (::realpath(binary_path.c_str(), absolute_path) == nullptr) {
      continue;
    }
    struct stat statinfo;
    int result = ::stat(absolute_path, &statinfo);
    if (result < 0) {
      continue;
    }
    if (!S_ISREG(statinfo.st_mode)) {
      continue;
    }

    // Is the current user able to execute the file?
    if (statinfo.st_uid == ::geteuid() && statinfo.st_mode & S_IXUSR) {
      return true;
    }
    // Is the current group able to execute the file?
    if (statinfo.st_uid == ::getegid() && statinfo.st_mode & S_IXGRP) {
      return true;
    }
    // Is anyone able to execute the file?
    if (statinfo.st_mode & S_IXOTH) {
      return true;
    }
  }
  return false;
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

// Reads a PCM file using signed little endian 16-bit encoding (s16le).
std::vector<float> ReadPcmFile(const string& filename) {
  string raw_data;
  TF_QCHECK_OK(ReadFileToString(Env::Default(), filename, &raw_data))
      << "Could not read FFmpeg output file: " << filename;

  std::vector<float> samples;
  const int32 sample_count = raw_data.size() / sizeof(int16);
  samples.reserve(sample_count);

  for (int32 i = 0; i < sample_count; ++i) {
    // Most of this is jumping through hoops in the standard to convert some
    // bits into the right format. I hope that an optimizing compiler will
    // remove almost all of this code.
    char raw[2] = {raw_data[i * 2], raw_data[i * 2 + 1]};
    if (!port::kLittleEndian) {
      std::swap(raw[0], raw[1]);
    }
    int16 host_order;
    ::memcpy(&host_order, raw, sizeof(host_order));
    const double normalized =
        static_cast<double>(host_order) / std::numeric_limits<int16>::max();
    samples.push_back(normalized);
  }
  return samples;
}

template <typename UInt>
string LittleEndianData(UInt data) {
  static_assert(std::is_unsigned<UInt>::value, "UInt must be unsigned");
  string str;
  for (size_t i = 0; i < sizeof(UInt); ++i) {
    const unsigned char bits = static_cast<unsigned char>(data & 0xFFU);
    char ch;
    ::memcpy(&ch, &bits, sizeof(bits));
    str.push_back(ch);
    data >>= 8;
  }
  return str;
}

string LittleEndianDataInt(uint32 data) {
  return LittleEndianData<uint32>(data);
}

string LittleEndianDataShort(uint16 data) {
  return LittleEndianData<uint16>(data);
}

string WavHeader(int32 samples_per_second, int32 channel_count,
                 const std::vector<float>& samples) {
  string header = "RIFF";
  header += LittleEndianDataInt(36U + samples.size() * sizeof(int16));
  header += "WAVEfmt ";
  header += LittleEndianDataInt(16);
  header += LittleEndianDataShort(1);
  header += LittleEndianDataShort(channel_count);
  header += LittleEndianDataInt(samples_per_second);
  header +=
      LittleEndianDataInt(samples_per_second * channel_count * sizeof(int16));
  header += LittleEndianDataShort(channel_count * sizeof(int16));
  header += LittleEndianDataShort(16);
  header += "data";
  header += LittleEndianDataInt(samples.size() * sizeof(int16));
  CHECK_EQ(header.size(), 44);
  return header;
}

// Creates the contents of a .wav file using pcm_s16le format (signed 16 bit
// little endian integers).
string BuildWavFile(int32 samples_per_second, int32 channel_count,
                    const std::vector<float>& samples) {
  string data = WavHeader(samples_per_second, channel_count, samples);
  data.reserve(data.size() + samples.size() * sizeof(int16));
  for (float value : samples) {
    const int16 quantized =
        static_cast<int16>(value * std::numeric_limits<int16>::max());
    char raw[2];
    ::memcpy(raw, &quantized, sizeof(int16));
    if (!port::kLittleEndian) {
      std::swap(raw[0], raw[1]);
    }
    data.push_back(raw[0]);
    data.push_back(raw[1]);
  }
  return data;
}

// Returns a unique number every time it is called.
int64 UniqueId() {
  static mutex mu(LINKER_INITIALIZED);
  static int64 id = 0;
  mutex_lock l(mu);
  return ++id;
}

}  // namespace

string GetTempFilename(const string& extension) {
  for (const char* dir : std::vector<const char*>(
           {getenv("TEST_TMPDIR"), getenv("TMPDIR"), getenv("TMP"), "/tmp"})) {
    if (!dir || !dir[0]) {
      continue;
    }
    struct stat statbuf;
    if (!stat(dir, &statbuf) && S_ISDIR(statbuf.st_mode)) {
      // UniqueId is added here because mkstemps is not as thread safe as it
      // looks. https://github.com/tensorflow/tensorflow/issues/5804 shows
      // the problem.
      string tmp_filepath = io::JoinPath(
          dir,
          StrCat("tmp_file_tensorflow_", UniqueId(), "_XXXXXX.", extension));
      int fd = mkstemps(&tmp_filepath[0], extension.length() + 1);
      if (fd < 0) {
        LOG(FATAL) << "Failed to create temp file.";
      } else {
        close(fd);
        return tmp_filepath;
      }
    }
  }
  LOG(FATAL) << "No temp directory found.";
}

Status ReadAudioFile(const string& filename,
                     const string& audio_format_id,
                     int32 samples_per_second,
                     int32 channel_count,
                     std::vector<float>* output_samples) {
  // Create an argument list.
  string output_filename = GetTempFilename("raw");
  const std::vector<string> args =
      FfmpegCommandLine(filename, output_filename, audio_format_id,
                        samples_per_second, channel_count);

  // Unfortunately, it's impossible to differentiate an exec failure due to the
  // binary being missing and an error from the binary's execution. Therefore,
  // check to see if the binary *should* be available. If not, return an error
  // that will be converted into a helpful error message by the TensorFlow op.
  if (!IsBinaryInstalled(kFfmpegExecutable)) {
    return Status(error::Code::NOT_FOUND, StrCat("FFmpeg could not be found."));
  }

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
    if (status_code) {
      return Status(error::Code::UNKNOWN,
                    StrCat("FFmpeg execution failed: ", status_code));
    }
    *output_samples = ReadPcmFile(output_filename);
    TF_QCHECK_OK(Env::Default()->DeleteFile(output_filename))
        << output_filename;
    return Status::OK();
  }
}

Status CreateAudioFile(const string& audio_format_id, int32 bits_per_second,
                       int32 samples_per_second, int32 channel_count,
                       const std::vector<float>& samples, string* output_data) {
  if (audio_format_id != "wav") {
    return Status(error::Code::INVALID_ARGUMENT,
                  "CreateAudioFile only supports the 'wav' audio format.");
  }
  *output_data = BuildWavFile(samples_per_second, channel_count, samples);
  return Status::OK();
}

}  // namespace ffmpeg
}  // namespace tensorflow
