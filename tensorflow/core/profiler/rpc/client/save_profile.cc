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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/utils/file_system_utils.h"

// Windows.h #defines ERROR, but it is also used in
// tensorflow/core/util/event.proto
#undef ERROR
#include "tensorflow/core/util/events_writer.h"

namespace tensorflow {
namespace profiler {
namespace {


constexpr char kProtoTraceFileName[] = "trace";
constexpr char kTfStatsHelperSuffix[] = "tf_stats_helper_result";

Status DumpToolData(absl::string_view run_dir, absl::string_view host,
                    const ProfileToolData& tool, std::ostream* os) {
  // Don't save the intermediate results for combining the per host tool data.
  if (absl::EndsWith(tool.name(), kTfStatsHelperSuffix)) return Status::OK();
  std::string host_prefix = host.empty() ? "" : absl::StrCat(host, ".");
  std::string path =
      ProfilerJoinPath(run_dir, absl::StrCat(host_prefix, tool.name()));
  TF_RETURN_IF_ERROR(WriteStringToFile(Env::Default(), path, tool.data()));
  if (os) {
    *os << "Dumped tool data for " << tool.name() << " to " << path << '\n';
  }
  return Status::OK();
}

Status WriteGzippedDataToFile(const std::string& filepath,
                              const std::string& data) {
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

Status GetOrCreateRunDir(const std::string& repository_root,
                         const std::string& run, std::string* run_dir,
                         std::ostream* os) {
  // Creates a directory to <repository_root>/<run>/.
  *run_dir = ProfilerJoinPath(repository_root, run);
  *os << "Creating directory: " << *run_dir << '\n';
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(*run_dir));
  return Status::OK();
}
}  // namespace

std::string GetTensorBoardProfilePluginDir(const std::string& logdir) {
  constexpr char kPluginName[] = "plugins";
  constexpr char kProfileName[] = "profile";
  return ProfilerJoinPath(logdir, kPluginName, kProfileName);
}

Status MaybeCreateEmptyEventFile(const std::string& logdir) {
  // Suffix for an empty event file.  it should be kept in sync with
  // _EVENT_FILE_SUFFIX in tensorflow/python/eager/profiler.py.
  constexpr char kProfileEmptySuffix[] = ".profile-empty";
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(logdir));

  std::vector<std::string> children;
  TF_RETURN_IF_ERROR(Env::Default()->GetChildren(logdir, &children));
  for (const std::string& child : children) {
    if (absl::EndsWith(child, kProfileEmptySuffix)) {
      return Status::OK();
    }
  }
  EventsWriter event_writer(ProfilerJoinPath(logdir, "events"));
  return event_writer.InitWithSuffix(kProfileEmptySuffix);
}

Status SaveProfile(const std::string& repository_root, const std::string& run,
                   const std::string& host, const ProfileResponse& response,
                   std::ostream* os) {
  if (response.tool_data().empty()) return Status::OK();
  std::string run_dir;
  TF_RETURN_IF_ERROR(GetOrCreateRunDir(repository_root, run, &run_dir, os));
  // Windows file names do not support colons.
  std::string hostname = absl::StrReplaceAll(host, {{":", "_"}});
  for (const auto& tool_data : response.tool_data()) {
    TF_RETURN_IF_ERROR(DumpToolData(run_dir, hostname, tool_data, os));
  }
  return Status::OK();
}

Status SaveGzippedToolData(const std::string& repository_root,
                           const std::string& run, const std::string& host,
                           const std::string& tool_name,
                           const std::string& data) {
  std::string run_dir;
  std::stringstream ss;
  Status status = GetOrCreateRunDir(repository_root, run, &run_dir, &ss);
  LOG(INFO) << ss.str();
  TF_RETURN_IF_ERROR(status);
  std::string host_prefix = host.empty() ? "" : absl::StrCat(host, ".");
  std::string path =
      ProfilerJoinPath(run_dir, absl::StrCat(host_prefix, tool_name));
  TF_RETURN_IF_ERROR(WriteGzippedDataToFile(path, data));
  LOG(INFO) << "Dumped gzipped tool data for " << tool_name << " to " << path;
  return Status::OK();
}

std::string GetCurrentTimeStampAsString() {
  return absl::FormatTime("%E4Y_%m_%d_%H_%M_%S", absl::Now(),
                          absl::LocalTimeZone());
}

}  // namespace profiler
}  // namespace tensorflow
