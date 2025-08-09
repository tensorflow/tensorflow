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

#include "xla/tsl/profiler/rpc/client/save_profile.h"

#include <memory>
#include <ostream>
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
#include "xla/tsl/lib/io/zlib_compression_options.h"
#include "xla/tsl/lib/io/zlib_outputbuffer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

constexpr char kProtoTraceFileName[] = "trace";
constexpr char kTfStatsHelperSuffix[] = "tf_stats_helper_result";
constexpr char kXPlanePb[] = "xplane.pb";

absl::Status DumpToolData(absl::string_view run_dir, absl::string_view host,
                          const tensorflow::ProfileToolData& tool,
                          std::ostream* os) {
  // Don't save the intermediate results for combining the per host tool data.
  if (absl::EndsWith(tool.name(), kTfStatsHelperSuffix)) {
    return absl::OkStatus();
  }
  std::string host_prefix = host.empty() ? "" : absl::StrCat(host, ".");
  std::string path =
      ProfilerJoinPath(run_dir, absl::StrCat(host_prefix, tool.name()));
  TF_RETURN_IF_ERROR(WriteStringToFile(Env::Default(), path, tool.data()));
  if (os) {
    *os << "Dumped tool data for " << tool.name() << " to " << path << '\n';
  }
  return absl::OkStatus();
}

absl::Status WriteGzippedDataToFile(const std::string& filepath,
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
  return absl::OkStatus();
}

absl::Status GetOrCreateRunDir(const std::string& repository_root,
                               const std::string& run, std::string* run_dir,
                               std::ostream* os) {
  // Creates a directory to <repository_root>/<run>/.
  *run_dir = ProfilerJoinPath(repository_root, run);
  *os << "Creating directory: " << *run_dir << '\n';
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(*run_dir));
  return absl::OkStatus();
}
}  // namespace

std::string GetTensorBoardProfilePluginDir(const std::string& logdir) {
  constexpr char kPluginName[] = "plugins";
  constexpr char kProfileName[] = "profile";
  return ProfilerJoinPath(logdir, kPluginName, kProfileName);
}

absl::Status SaveProfile(const std::string& repository_root,
                         const std::string& run, const std::string& host,
                         const tensorflow::ProfileResponse& response,
                         std::ostream* os) {
  if (response.tool_data().empty()) {
    return absl::OkStatus();
  }
  std::string run_dir;
  TF_RETURN_IF_ERROR(GetOrCreateRunDir(repository_root, run, &run_dir, os));
  // Windows file names do not support colons.
  std::string hostname = absl::StrReplaceAll(host, {{":", "_"}});
  for (const auto& tool_data : response.tool_data()) {
    TF_RETURN_IF_ERROR(DumpToolData(run_dir, hostname, tool_data, os));
  }
  return absl::OkStatus();
}

absl::Status SaveGzippedToolData(const std::string& repository_root,
                                 const std::string& run,
                                 const std::string& host,
                                 const std::string& tool_name,
                                 const std::string& data) {
  std::string run_dir;
  std::stringstream ss;
  absl::Status status = GetOrCreateRunDir(repository_root, run, &run_dir, &ss);
  LOG(INFO) << ss.str();
  TF_RETURN_IF_ERROR(status);
  std::string host_prefix = host.empty() ? "" : absl::StrCat(host, ".");
  std::string path =
      ProfilerJoinPath(run_dir, absl::StrCat(host_prefix, tool_name));
  TF_RETURN_IF_ERROR(WriteGzippedDataToFile(path, data));
  LOG(INFO) << "Dumped gzipped tool data for " << tool_name << " to " << path;
  return absl::OkStatus();
}

std::string GetCurrentTimeStampAsString() {
  return absl::FormatTime("%E4Y_%m_%d_%H_%M_%S", absl::Now(),
                          absl::LocalTimeZone());
}

absl::Status SaveXSpace(const std::string& repository_root,
                        const std::string& run, const std::string& host,
                        const tensorflow::profiler::XSpace& xspace) {
  std::string log_dir = ProfilerJoinPath(repository_root, run);
  VLOG(1) << "Creating " << log_dir;
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(log_dir));
  std::string file_name = absl::StrCat(host, ".", kXPlanePb);
  // Windows file names do not support colons.
  absl::StrReplaceAll({{":", "_"}}, &file_name);

  // Dumps profile data to <repository_root>/<run>/<host>_<port>.<kXPlanePb>
  std::string out_path = ProfilerJoinPath(log_dir, file_name);
  LOG(INFO) << "Collecting XSpace to repository: " << out_path;

  return WriteBinaryProto(Env::Default(), out_path, xspace);
}

}  // namespace profiler
}  // namespace tsl
