/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/records/record_reader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"  // IWYU pragma: keep
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::absl_testing::IsOk;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::testing::ElementsAre;
using ::testing::Not;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using ::tsl::proto_testing::Partially;

TEST(SaveProfileTest, SaveXSpaceChunksVectorSuccessAndPadding) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_batch_vector";
  std::string host = "test_host_batch_vector";

  XSpace space1 = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "host1"
    planes { name: "plane1" }
  )pb");

  XSpace space2 = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "host2"
    planes { name: "plane2" }
  )pb");

  std::vector<XSpace> spaces = {space1, space2};
  ASSERT_OK(SaveXSpaceChunks(temp_dir, run, host, spaces));
  EXPECT_TRUE(spaces.empty());

  std::string file_path =
      io::JoinPath(temp_dir, run, "test_host_batch_vector.xplane.riegeli");
  EXPECT_OK(Env::Default()->FileExists(file_path));

  uint64_t file_size = 0;
  ASSERT_OK(Env::Default()->GetFileSize(file_path, &file_size));
  EXPECT_EQ(file_size % (64 * 1024), 0);

  std::vector<XSpace> read_spaces;
  riegeli::RecordReader<riegeli::FdReader<>> reader{
      riegeli::FdReader<>(file_path)};
  XSpace read_space;
  while (reader.ReadRecord(read_space)) {
    read_spaces.push_back(read_space);
    read_space.Clear();
  }
  ASSERT_OK(reader.status());
  reader.Close();

  EXPECT_THAT(read_spaces, ElementsAre(Partially(EqualsProto(R"pb(
                                         hostnames: "host1"
                                         planes { name: "plane1" }
                                       )pb")),
                                       Partially(EqualsProto(R"pb(
                                         hostnames: "host2"
                                         planes { name: "plane2" }
                                       )pb"))));
}

TEST(SaveProfileTest, SaveXSpaceChunksEmptyVectorReturnsOkWithoutCreatingDir) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_empty_vector";
  std::string log_dir = io::JoinPath(temp_dir, run);
  std::vector<XSpace> empty_spaces;
  EXPECT_OK(SaveXSpaceChunks(temp_dir, run, "host", empty_spaces));
  EXPECT_THAT(Env::Default()->FileExists(log_dir), Not(IsOk()));
}

TEST(SaveProfileTest, SaveXSpaceChunksFailurePropagation) {
  std::string invalid_dir = "/proc/invalid_nonexistent_dir_for_test";
  std::string run = "test_run";
  std::string host = "test_host";

  XSpace space = ParseTextProtoOrDie<XSpace>(R"pb(
    hostnames: "host1"
  )pb");
  std::vector<XSpace> spaces = {space};

  EXPECT_THAT(SaveXSpaceChunks(invalid_dir, run, host, spaces), Not(IsOk()));
}
}  // namespace
}  // namespace profiler
}  // namespace tsl
