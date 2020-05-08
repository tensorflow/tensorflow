/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/rpc/client/save_profile.h"

#include <cstdio>
#include <ctime>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
// Windows.h #defines ERROR, but it is also used in
// tensorflow/core/util/event.proto
#undef ERROR
#include "tensorflow/core/util/events_writer.h"

namespace tensorflow {
namespace profiler {
namespace {

#ifdef PLATFORM_WINDOWS
const absl::string_view kPathSep = "\\";
#else
const absl::string_view kPathSep = "/";
#endif

string ProfilerJoinPathImpl(std::initializer_list<absl::string_view> paths) {
  string result;
  for (absl::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = string(path);
      continue;
    }

    path = absl::StripPrefix(path, kPathSep);
    if (absl::EndsWith(result, kPathSep)) {
      strings::StrAppend(&result, path);
    } else {
      strings::StrAppend(&result, kPathSep, path);
    }
  }

  return result;
}

// A local duplication of ::tensorflow::io::JoinPath that supports windows.
// TODO(b/150699701): revert to use ::tensorflow::io::JoinPath when fixed.
template <typename... T>
string ProfilerJoinPath(const T&... args) {
  return ProfilerJoinPathImpl({args...});
}

constexpr char kProtoTraceFileName[] = "trace";
constexpr char kTfStatsHelperSuffix[] = "tf_stats_helper_result";

Status DumpToolDataToLogDirectory(StringPiece run_dir, const string& host,
                                  const ProfileToolData& tool,
                                  std::ostream* os) {
  // Don't save the intermediate results for combining the per host tool data.
  if (absl::EndsWith(tool.name(), kTfStatsHelperSuffix)) return Status::OK();
  string host_prefix = host.empty() ? "" : absl::StrCat(host, ".");
  string path =
      ProfilerJoinPath(run_dir, absl::StrCat(host_prefix, tool.name()));
  TF_RETURN_IF_ERROR(WriteStringToFile(Env::Default(), path, tool.data()));
  if (os) {
    *os << "Dumped tool data for " << tool.name() << " to " << path
        << std::endl;
  }
  return Status::OK();
}

// Creates an empty event file if not already exists, which indicates that we
// have a plugins/profile/ directory in the current logdir.
Status MaybeCreateEmptyEventFile(const string& logdir) {
  // Suffix for an empty event file.  it should be kept in sync with
  // _EVENT_FILE_SUFFIX in tensorflow/python/eager/profiler.py.
  constexpr char kProfileEmptySuffix[] = ".profile-empty";
  std::vector<string> children;
  TF_RETURN_IF_ERROR(Env::Default()->GetChildren(logdir, &children));
  for (const string& child : children) {
    if (absl::EndsWith(child, kProfileEmptySuffix)) {
      return Status::OK();
    }
  }
  EventsWriter event_writer(ProfilerJoinPath(logdir, "events"));
  return event_writer.InitWithSuffix(kProfileEmptySuffix);
}

Status WriteGzippedDataToFile(const string& filepath, const string& data) {
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(filepath, &file));
  io::ZlibCompressionOptions options = io::ZlibCompressionOptions::GZIP();
  io::ZlibOutputBuffer buffer(file.get(), options.input_buffer_size,
                              options.output_buffer_size, options);
  TF_RETURN_IF_ERROR(buffer.Init());
  TF_RETURN_IF_ERROR(buffer.Append(data));
  TF_RETURN_IF_ERROR(buffer.Close());
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

Status GetOrCreateProfileRunDir(const string& logdir, const string& run,
                                string* profile_run_dir, std::ostream* os) {
  // Dumps profile data to <logdir>/plugins/profile/<run>/.
  *profile_run_dir =
      ProfilerJoinPath(GetTensorBoardProfilePluginDir(logdir), run);
  *os << "Creating directory: " << *profile_run_dir;
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(*profile_run_dir));

  // Creates an empty event file so that TensorBoard plugin logic can find
  // the logdir.
  TF_RETURN_IF_ERROR(MaybeCreateEmptyEventFile(logdir));
  return Status::OK();
}

}  // namespace

string GetTensorBoardProfilePluginDir(const string& logdir) {
  constexpr char kPluginName[] = "plugins";
  constexpr char kProfileName[] = "profile";
  return ProfilerJoinPath(logdir, kPluginName, kProfileName);
}

Status SaveTensorboardProfile(const string& logdir, const string& run,
                              const string& host,
                              const ProfileResponse& response,
                              std::ostream* os) {
  string profile_run_dir;
  TF_RETURN_IF_ERROR(
      GetOrCreateProfileRunDir(logdir, run, &profile_run_dir, os));
  for (const auto& tool_data : response.tool_data()) {
    TF_RETURN_IF_ERROR(
        DumpToolDataToLogDirectory(profile_run_dir, host, tool_data, os));
  }
  return Status::OK();
}

Status SaveGzippedToolDataToTensorboardProfile(const string& logdir,
                                               const string& run,
                                               const string& host,
                                               const string& tool_name,
                                               const string& data) {
  string profile_run_dir;
  std::stringstream ss;
  Status status = GetOrCreateProfileRunDir(logdir, run, &profile_run_dir, &ss);
  LOG(INFO) << ss.str();
  TF_RETURN_IF_ERROR(status);
  string host_prefix = host.empty() ? "" : absl::StrCat(host, ".");
  string path =
      ProfilerJoinPath(profile_run_dir, absl::StrCat(host_prefix, tool_name));
  TF_RETURN_IF_ERROR(WriteGzippedDataToFile(path, data));
  LOG(INFO) << "Dumped gzipped tool data for " << tool_name << " to " << path;
  return Status::OK();
}

string GetCurrentTimeStampAsString() {
  return absl::FormatTime("%E4Y_%m_%d_%H_%M_%S", absl::Now(),
                          absl::LocalTimeZone());
}

}  // namespace profiler
}  // namespace tensorflow
