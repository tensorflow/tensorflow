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
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <vector>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace ffmpeg {
namespace {

const char kFfmpegExecutable[] = "ffmpeg";
const int32 kDefaultProbeSize = 5000000;  // 5MB

std::vector<string> FfmpegAudioCommandLine(const string& input_filename,
                                           const string& output_filename,
                                           const string& input_format_id,
                                           int32 samples_per_second,
                                           int32 channel_count) {
  return {"-nostats",             // No additional progress display.
          "-nostdin",             // No interactive commands accepted.
          "-f", input_format_id,  // eg: "mp3"
          "-probesize", StrCat(kDefaultProbeSize), "-i", input_filename,
          "-loglevel", "error",   // Print errors only.
          "-hide_banner",         // Skip printing build options, version, etc.
          "-map_metadata", "-1",  // Copy global metadata from input to output.
          "-vn",                  // No video recording.
          "-ac:a:0", StrCat(channel_count), "-ar:a:0",
          StrCat(samples_per_second),
          // Output set (in several ways) to signed 16-bit little-endian ints.
          "-codec:a:0", "pcm_s16le", "-sample_fmt", "s16", "-f", "s16le",
          "-sn",  // No subtitle recording.
          "-y",   // Overwrite output file.
          StrCat(output_filename)};
}

std::vector<string> FfmpegVideoCommandLine(const string& input_filename,
                                           const string& output_filename) {
  return {"-nostats",  // No additional progress display.
          "-nostdin",  // No interactive commands accepted.
          "-i",
          input_filename,
          "-f",
          "image2pipe",
          "-probesize",
          StrCat(kDefaultProbeSize),
          "-loglevel",
          "error",  // Print errors only.
          "-hide_banner",  // Skip printing build options, version, etc.
          "-vcodec",
          "rawvideo",
          "-pix_fmt",
          "rgb24",
          "-y",  // Overwrite output file.
          StrCat(output_filename)};
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
  LOG(ERROR) << "FFmpeg could not be executed: " << strerror(error);
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

Status ReadInfoFile(const string& filename, uint32* width, uint32* height,
                    uint32* frames) {
  string data;
  TF_QCHECK_OK(ReadFileToString(Env::Default(), filename, &data))
      << "Could not read FFmpeg file: " << filename;
  bool in_output = false;
  bool in_mapping = false;
  uint32 frames_value = 0;
  uint32 height_value = 0;
  uint32 width_value = 0;
  for (const string& line : str_util::Split(data, '\n')) {
    // Output starts with the first line of `Output #..`.
    // Further processing output region starts next line so we could continue
    // the loop.
    if (!in_output && line.find("Output #") == 0) {
      in_output = true;
      in_mapping = false;
      continue;
    }
    // Stream mapping starts with the first line of `Stream mapping`, it also
    // signals the end of Output section.
    // Further processing of stream mapping region starts next line so we could
    // continue the loop.
    if (!in_mapping && line.find("Stream mapping:") == 0) {
      in_output = false;
      in_mapping = true;
      continue;
    }
    if (in_output) {
      // We only look for the first stream in output `Stream #0`.
      // Once processed we will not further process output section.
      if (line.find("    Stream #") == 0) {
        size_t p = line.find(", rgb24, ", 24);
        if (p != std::string::npos) {
          string rgb24 = line.substr(p + 9, line.find(" ", p + 9));
          rgb24 = rgb24.substr(0, rgb24.find(","));
          string rgb24_width = rgb24.substr(0, rgb24.find("x"));
          string rgb24_height = rgb24.substr(rgb24_width.length() + 1);
          if (strings::safe_strtou32(rgb24_width, &width_value) &&
              strings::safe_strtou32(rgb24_height, &height_value)) {
            in_output = false;
          }
        }
      }
      continue;
    }
    if (in_mapping) {
      // We only look for the first stream mapping to have the number of the
      // frames.
      // Once processed we will not further process stream mapping section.
      if (line.find("frame=  ") == 0) {
        string number = line.substr(8, line.find(" ", 8));
        number = number.substr(0, number.find(" "));
        if (strings::safe_strtou32(number, &frames_value)) {
          in_mapping = false;
        }
      }
      continue;
    }
  }
  if (frames_value == 0 || height_value == 0 || width_value == 0) {
    return errors::Unknown("Not enough video info returned by FFmpeg [",
                           frames_value, ", ", height_value, ", ", width_value,
                           ", 3]");
  }
  *width = width_value;
  *height = height_value;
  *frames = frames_value;
  return Status::OK();
}

}  // namespace

FileDeleter::~FileDeleter() {
  Env& env = *Env::Default();
  env.DeleteFile(filename_).IgnoreError();
}

Status WriteFile(const string& filename, StringPiece contents) {
  Env& env = *Env::Default();
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env.NewWritableFile(filename, &file));
  TF_RETURN_IF_ERROR(file->Append(contents));
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

Status ReadAudioFile(const string& filename, const string& audio_format_id,
                     int32 samples_per_second, int32 channel_count,
                     std::vector<float>* output_samples) {
  // Create an argument list.
  string output_filename = io::GetTempFilename("raw");
  const std::vector<string> args =
      FfmpegAudioCommandLine(filename, output_filename, audio_format_id,
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
    return Status(error::Code::UNKNOWN,
                  StrCat("fork failed: ", strerror(errno)));
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

Status ReadVideoFile(const string& filename, std::vector<uint8>* output_data,
                     uint32* width, uint32* height, uint32* frames) {
  if (!IsBinaryInstalled(kFfmpegExecutable)) {
    return Status(error::Code::NOT_FOUND, StrCat("FFmpeg could not be found."));
  }

  string output_filename = io::GetTempFilename("raw");
  string stderr_filename = io::GetTempFilename("err");

  // Create an argument list.
  const std::vector<string> args =
      FfmpegVideoCommandLine(filename, output_filename);

  // Execute ffmpeg and report errors.
  pid_t child_pid = ::fork();
  if (child_pid < 0) {
    return Status(error::Code::UNKNOWN,
                  StrCat("fork failed: ", strerror(errno)));
  }
  if (child_pid == 0) {
    const int fd =
        open(stderr_filename.c_str(), O_RDWR | O_CREAT | O_APPEND, 0600);
    if (fd < 0) {
      const int error = errno;
      LOG(ERROR) << "FFmpeg stderr file could not be created: "
                 << strerror(error);
      ::_exit(error);
    }
    close(STDERR_FILENO);
    dup2(fd, STDERR_FILENO);
    ExecuteFfmpeg(args);
  } else {
    int status_code;
    if (::waitpid(child_pid, &status_code, 0) < 0) {
      return Status(error::Code::UNKNOWN,
                    StrCat("waitpid failed: ", strerror(errno)));
    }
    if (status_code) {
      return Status(error::Code::UNKNOWN,
                    StrCat("FFmpeg execution failed: ", status_code));
    }

    TF_QCHECK_OK(ReadInfoFile(stderr_filename, width, height, frames))
        << "Could not read FFmpeg stderr file: " << stderr_filename;

    string raw_data;
    TF_QCHECK_OK(ReadFileToString(Env::Default(), output_filename, &raw_data))
        << "Could not read FFmpeg output file: " << output_filename;
    output_data->resize(raw_data.size());
    std::copy_n(raw_data.data(), raw_data.size(), output_data->begin());

    TF_QCHECK_OK(Env::Default()->DeleteFile(output_filename))
        << output_filename;
    TF_QCHECK_OK(Env::Default()->DeleteFile(stderr_filename))
        << stderr_filename;
    return Status::OK();
  }
}
}  // namespace ffmpeg
}  // namespace tensorflow
