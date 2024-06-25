/* Copyright 2024 The OpenXLA Authors.

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

// Tool to process a HloModuleMetadataProto that was dumped with the flag
// --xla_dump_module_metadata
//
// Usage: hlo_module_metadata_processor <filepath>
// Where <filepath> should point to a file with a serialized HloModuleMetadata
// proto in text format.
//
// The tool writes the individual pass timings to stdout in the following
// format:
// Pass timings for <pass name>: id <id0>: x ms, id <id1>: y ms, ...
// The pass timinings of the individual runs of the same pass are sorted in
// non-increasing order based on runtime. Also, the different passes are sorted
// in non-increasing order based on the maximum runtime of a pass.
// The idea is that with this output, it is easier to spot the pass id of a pass
// with an unexpected high runtime. This pass id can then be looked up in the
// dump to gather additional data, e.g. which pipeline the pass was run in.

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/service/hlo.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"

namespace xla {
namespace tools {
namespace {
struct HloPassMetadataFormatter {
  void operator()(std::string* out, const HloPassMetadata& data) const {
    out->append(absl::StrFormat(
        "id %d: %d ms", data.pass_id(),
        (data.end_timestamp_usec() - data.start_timestamp_usec()) / 1000));
  }
};

void ProcessMetadata(const std::string& serialized) {
  HloModuleMetadataProto metadata;
  if (!tsl::protobuf::TextFormat::ParseFromString(serialized, &metadata)) {
    LOG(FATAL) << "Unable to parse HloModuleMetadata";
  }
  absl::flat_hash_map<std::string, std::vector<HloPassMetadata>>
      group_by_pass_name;
  for (const auto& pass_metadata : metadata.pass_metadata()) {
    group_by_pass_name[pass_metadata.pass_name()].push_back(pass_metadata);
  }
  std::vector<std::string> pass_names_sorted_by_time;
  for (auto& entry : group_by_pass_name) {
    pass_names_sorted_by_time.push_back(entry.first);
    std::sort(entry.second.begin(), entry.second.end(),
              [](const HloPassMetadata& a, const HloPassMetadata& b) {
                return a.end_timestamp_usec() - a.start_timestamp_usec() >
                       b.end_timestamp_usec() - b.start_timestamp_usec();
              });
  }
  std::sort(
      pass_names_sorted_by_time.begin(), pass_names_sorted_by_time.end(),
      [&](const std::string& a, const std::string& b) {
        const auto& a_data = group_by_pass_name[a][0];
        const auto& b_data = group_by_pass_name[b][0];
        return a_data.end_timestamp_usec() - a_data.start_timestamp_usec() >
               b_data.end_timestamp_usec() - b_data.start_timestamp_usec();
      });
  for (const auto& name : pass_names_sorted_by_time) {
    const auto& data = group_by_pass_name[name];
    std::cout << "Pass timings for " << name << ": "
              << absl::StrJoin(data, ", ", HloPassMetadataFormatter())
              << std::endl;
  }
}
}  // namespace
}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);

  QCHECK_EQ(argc, 2) << "usage: " << argv[0] << " <filepath>";
  std::string serialized;
  TF_CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), std::string(argv[1]),
                                    &serialized));
  xla::tools::ProcessMetadata(serialized);
  return 0;
}
