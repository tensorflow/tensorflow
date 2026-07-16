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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/records/record_reader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"  // IWYU pragma: keep
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::absl_testing::IsOk;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::testing::Not;

TEST(SaveProfileTest, SaveXSpaceChunksVectorSuccessAndPadding) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_batch_vector";
  std::string host = "test_host_batch_vector";

  XSpace space1;
  space1.add_hostnames("host1");
  XPlane* plane1 = space1.add_planes();
  plane1->set_name("plane1");

  XSpace space2;
  space2.add_hostnames("host2");
  XPlane* plane2 = space2.add_planes();
  plane2->set_name("plane2");

  std::vector<XSpace> spaces = {space1, space2};
  ASSERT_OK(SaveXSpaceChunks(temp_dir, run, host, &spaces));
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

  ASSERT_EQ(read_spaces.size(), 2);
  EXPECT_EQ(read_spaces[0].hostnames(0), "host1");
  EXPECT_EQ(read_spaces[0].planes(0).name(), "plane1");
  EXPECT_EQ(read_spaces[1].hostnames(0), "host2");
  EXPECT_EQ(read_spaces[1].planes(0).name(), "plane2");

  ASSERT_OK(Env::Default()->DeleteFile(file_path));
}

TEST(SaveProfileTest, SaveXSpaceChunksEmptyVectorReturnsOkWithoutCreatingDir) {
  std::string temp_dir = testing::TmpDir();
  std::string run = "test_run_empty_vector";
  std::string log_dir = io::JoinPath(temp_dir, run);
  std::vector<XSpace> empty_spaces;
  EXPECT_OK(SaveXSpaceChunks(temp_dir, run, "host", &empty_spaces));
  EXPECT_THAT(Env::Default()->FileExists(log_dir), Not(IsOk()));
  EXPECT_OK(SaveXSpaceChunks(temp_dir, run, "host", nullptr));
}

TEST(SaveProfileTest, SaveXSpaceChunksInvalidDirectoryDoesNotLeaveTempFile) {
  std::string temp_dir = testing::TmpDir();
  std::string file_as_dir = io::JoinPath(temp_dir, "file_as_dir_spaces");
  ASSERT_OK(WriteStringToFile(Env::Default(), file_as_dir, "content"));

  XSpace space;
  std::vector<XSpace> spaces = {space};
  absl::Status status =
      SaveXSpaceChunks(file_as_dir, "subdir", "host", &spaces);
  EXPECT_THAT(status, Not(IsOk()));

  std::string expected_out =
      io::JoinPath(file_as_dir, "subdir", "host.xplane.riegeli");
  EXPECT_THAT(Env::Default()->FileExists(expected_out), Not(IsOk()));
  ASSERT_OK(Env::Default()->DeleteFile(file_as_dir));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
