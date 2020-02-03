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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
// Windows.h #defines ERROR, but it is also used in
// tensorflow/core/util/event.proto
#undef ERROR
#include "tensorflow/core/protobuf/trace_events.pb.h"
#include "tensorflow/core/util/events_writer.h"

namespace tensorflow {

namespace profiler {
namespace {

using ::tensorflow::io::JoinPath;

constexpr char kProfilePluginDirectory[] = "plugins/profile/";
constexpr char kProtoTraceFileName[] = "trace";

constexpr char kTfStatsHelperSuffix[] = "tf_stats_helper_result";

Status DumpTraceToLogDirectory(StringPiece run_dir, const string& host_prefix,
                               const string& encoded_trace, std::ostream* os) {
  string proto_path =
      JoinPath(run_dir, absl::StrCat(host_prefix, kProtoTraceFileName));
  TF_RETURN_IF_ERROR(
      WriteStringToFile(Env::Default(), proto_path, encoded_trace));
  if (os) *os << "Dumped raw-proto trace data to " << proto_path;
  return Status::OK();
}

Status DumpToolDataToLogDirectory(StringPiece run_dir,
                                  const string& host_prefix,
                                  const ProfileToolData& tool,
                                  std::ostream* os) {
  // Don't save the intermediate results for combining the per host tool data.
  if (absl::EndsWith(tool.name(), kTfStatsHelperSuffix)) return Status::OK();
  string path = JoinPath(run_dir, absl::StrCat(host_prefix, tool.name()));
  TF_RETURN_IF_ERROR(WriteStringToFile(Env::Default(), path, tool.data()));
  if (os) {
    *os << "Dumped tool data for " << tool.name() << " to " << path
        << std::endl;
  }
  return Status::OK();
}

}  // namespace

Status SaveTensorboardProfile(const string& logdir, const string& run,
                              const string& host,
                              const ProfileResponse& response,
                              std::ostream* os) {
  // Dumps profile data to <logdir>/plugins/profile/<run>/.
  string host_prefix = host.empty() ? "" : absl::StrCat(host, ".");
  string profile_run_dir = JoinPath(logdir, kProfilePluginDirectory, run);
  *os << "Creating directory: " << profile_run_dir;
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(profile_run_dir));

  // Ignore computation_graph for now.
  if (!response.encoded_trace().empty()) {
    TF_RETURN_IF_ERROR(DumpTraceToLogDirectory(profile_run_dir, host_prefix,
                                               response.encoded_trace(), os));
  }
  for (const auto& tool_data : response.tool_data()) {
    TF_RETURN_IF_ERROR(DumpToolDataToLogDirectory(profile_run_dir, host_prefix,
                                                  tool_data, os));
  }

  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
